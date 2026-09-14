"""Shared scoring utilities for client-side and server-side evaluation."""

import copy

import numpy as np
import torch
import torch.nn.functional as F


def compute_group_cardinality(group_name, sensitive_attributes):
    """
    Compute the number of possible combinations for a sensitive group.

    Args:
        group_name: Name of the sensitive/group attribute.
        sensitive_attributes: Iterable of ``(name, values)`` group metadata.

    Returns:
        Product of the cardinalities of the group's component attributes.

    Raises:
        KeyError: If ``group_name`` is not present in ``sensitive_attributes``.
    """
    for name, group_dict in sensitive_attributes:
        if name == group_name:
            total = 1
            for values in group_dict.values():
                total *= len(values)
            return total
    raise KeyError(f"Group {group_name} not found in sensitive attributes")


def average_dictionary_list(dictionary_list):
    """
    Average dictionaries that share the same numeric keys.

    Args:
        dictionary_list: Non-empty list of dictionaries with numeric values.

    Returns:
        Dictionary with the same keys and averaged values.
    """
    result = {key: 0 for key in dictionary_list[0].keys()}
    for dictionary in dictionary_list:
        for key, value in dictionary.items():
            result[key] += value
    for key in result:
        result[key] /= len(dictionary_list)
    return result


def scoring_function(results, use_training=False, weight_constraint=1, split=None):
    """
    Compute the scalar score used to rank candidate models.

    The score starts from the predictive objective, subtracts the optional
    FairLAB performance-budget penalty, and then subtracts fairness violations.

    Args:
        results: Aggregated evaluation dictionary.
        use_training: Whether to use training metrics instead of validation.
        weight_constraint: Multiplier applied to each constraint violation.

    Returns:
        Scalar score where larger values are better.
    """
    prefix = split or ("train" if use_training else "val")
    score = results[f"{prefix}_objective_fn"]
    score -= (
        results.get(f"{prefix}_performance_penalty", 0.0)
        * weight_constraint
    )
    for constraint in results[f"{prefix}_constraints"]:
        score -= constraint * weight_constraint
    return score


def collect_local_results(**kwargs):
    """
    Build per-client local score summaries for candidate models.

    Args:
        **kwargs: Requires ``eval_results``, ``model_params`` and
            ``original_threshold_list``.

    Returns:
        Nested dictionary indexed by evaluator client and model index.
    """
    results = kwargs.get("eval_results")
    params = kwargs.get("model_params")
    thresholds = list(kwargs.get("original_threshold_list") or ())
    assert len(results) == len(params), "Results and parameters must have the same length"

    local_results = {i: {} for i in range(len(results))}
    for evaluator_idx, evaluator_results in enumerate(results):
        for model_idx, result in enumerate(evaluator_results):
            thresholded_constraints = [
                max(0, constraint - thresholds[idx])
                for idx, constraint in enumerate(result["val_constraints"])
            ]
            local_result = copy.deepcopy(result)
            local_result["val_constraints"] = thresholded_constraints
            local_result["metrics"]["val_constraints_score"] = scoring_function(
                local_result,
                use_training=False,
            )
            local_results[evaluator_idx][model_idx] = {
                "model_params": params[evaluator_idx],
                "score": local_result["metrics"]["val_constraints_score"],
            }

    return local_results


def select_from_scores(scores, tau=1):
    """
    Sample an index from scores while also returning the deterministic minimum.

    Args:
        scores: Scores used for selection or probability computation.
        tau: Softmax temperature.

    Returns:
        Tuple ``(sampled_index, argmin_index)``.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores * 10
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    probabilities = F.softmax(-scores / tau, dim=0)
    selected = torch.multinomial(probabilities, num_samples=1).item()
    argmin = torch.argmin(scores).item()
    return selected, argmin


def compute_global_score(**kwargs):
    """
    Aggregate client evaluations into the global FedFairLab score.

    Args:
        **kwargs: Requires ``eval_results``. Optional keys include
    ``performance_constraint``, ``performance_reference`` and
            ``original_threshold_list``.

    Returns:
        Aggregated evaluation dictionary with ``metrics['val_global_score']``.
    """
    split = kwargs.get("split", "val")
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    performance_constraint = kwargs.get("performance_constraint")
    performance_reference = kwargs.get("performance_reference", None)
    original_threshold_list = tuple(kwargs.get("original_threshold_list") or ())
    results = kwargs.get("eval_results")
    assert results is not None, "Evaluation results are required"

    global_scores = _average_eval_results(results)
    _apply_fairness_thresholds(global_scores, original_threshold_list, split=split)
    _apply_performance_budget(
        global_scores,
        performance_constraint=performance_constraint,
        performance_reference=performance_reference,
        split=split,
    )
    for suffix in (
        "performance_reference",
        "performance_target",
        "performance_penalty",
    ):
        key = f"{split}_{suffix}"
        global_scores["metrics"][key] = global_scores[key]

    global_scores["metrics"].pop(f"{split}_constraints_score", None)
    global_scores["metrics"][f"{split}_global_score"] = scoring_function(
        global_scores,
        use_training=False,
        weight_constraint=kwargs.get("constraint_weight", 1.0),
        split=split,
    )
    return global_scores


def _average_eval_results(results):
    """Average raw per-client evaluation dictionaries."""
    grouped_results = {key: [] for key in results[0].keys()}
    for result in results:
        for key, value in result.items():
            grouped_results[key].append(value)

    averaged = {}
    for key, values in grouped_results.items():
        if isinstance(values[0], dict):
            averaged[key] = average_dictionary_list(values)
        else:
            averaged[key] = np.mean(np.array(values), axis=0)
    return averaged


def _apply_fairness_thresholds(global_scores, original_threshold_list, split="val"):
    """Convert validation constraint values into non-negative violations."""
    thresholds = list(original_threshold_list)
    if not thresholds:
        return

    constraints_key = f"{split}_constraints"
    constraints = global_scores.get(constraints_key)
    if constraints is None:
        return

    global_scores[constraints_key] = [
        max(0, constraint - thresholds[idx])
        for idx, constraint in enumerate(constraints)
    ]


def _apply_performance_budget(
    global_scores,
    *,
    performance_constraint,
    performance_reference,
    split="val",
):
    """Attach the FairLAB performance-budget penalty to global scores."""
    if (
        performance_constraint is not None
        and performance_reference is not None
    ):
        performance_target = max(0.0, performance_reference - performance_constraint)
        global_scores[f"{split}_performance_penalty"] = max(
            0,
            performance_target - global_scores[f"{split}_objective_fn"],
        )
        global_scores[f"{split}_performance_target"] = performance_target
    else:
        global_scores[f"{split}_performance_penalty"] = 0.0
        global_scores[f"{split}_performance_target"] = None

    global_scores[f"{split}_performance_reference"] = performance_reference
