from debug_utils import debug_print
from .client_base import BaseClient
import ray
from .client_factory import register_client
from wrappers import OrchestratorWrapper
import copy
import numpy as np
import time
from .utils import compute_global_score
from callbacks import ModelCheckpoint
import torch
from functools import partial
from checkpoint_utils import load_trusted_checkpoint
import random


def _state_dict_to_cpu(state_dict):
    """
    Detach a model state dict from accelerator memory before Ray serialization.

    Args:
        state_dict: PyTorch state dictionary possibly containing CUDA tensors.

    Returns:
        CPU-backed copy safe to send through Ray object storage.
    """
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


def _prefix_final_local_metrics(validation_metrics, test_metrics):
    """Build backward-compatible and explicit final local metric names."""
    final_metrics = {
        f'final_{key}': value
        for key, value in validation_metrics.items()
    }
    final_metrics.update({
        f'final_local_{key}': value
        for key, value in validation_metrics.items()
    })
    final_metrics.update({
        f'final_local_{key}': value
        for key, value in test_metrics.items()
    })
    return final_metrics


@register_client("client_fedfairlab")
@ray.remote(num_cpus=2)
class ClientFedFairLab(BaseClient):
    """
    Ray actor implementing a FedFairLab client.

    The client owns its local data module through an ``OrchestratorWrapper`` and
    exposes RPC methods used by the server for local mitigation, candidate
    evaluation, ensemble-logit computation, and shutdown.
    """

    def profile(func):
        """
        Decorate client RPC methods with lightweight wall-clock profiling.

        Args:
            func: Method to wrap.

        Returns:
            Wrapped method that prints execution time with the client name.
        """
        def wrapper(*args, **kwargs):
            """Execute the wrapped method and print its runtime."""
            self = args[0] if args else None
            name = getattr(self, 'client_name', 'Unknown')

            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            debug_print(
                f"[PROFILE {name}] {func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper

    def __init__(self, **kwargs):
        """
        Initialize a FedFairLab client actor.

        Args:
            **kwargs: Client configuration. Required entries include
                ``client_name``, ``logger``, ``orchestrator``, ``model``,
                ``client_callbacks``, FairLAB orchestrator epochs, and local
                learner epochs.
        """
        # self.orchestrator = kwargs.get('orchestrator')
        config = kwargs.get('config', {})
        random_seed = config.get('random_seed')
        if random_seed is not None:
            random_seed = int(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

        self.client_name = kwargs.get('client_name')
        self.logger_fn = kwargs.get('logger')
        self.logger = self.logger_fn()
        self.orchestrator_fn = kwargs.get('orchestrator')
        self.num_global_epochs = kwargs.get('num_global_iterations', 1)
        self.num_local_epochs = kwargs.get('num_local_iterations', 1)

        model = kwargs.get('model')
        self.client_checkpoints = kwargs.get('client_callbacks')
        assert model is not None, "Model is required"
        self.state = None
        self.orchestrator_map = {}
        self.local_scoring_config = None
        self.performance_reference = None

        debug_print(f"Client {self.client_name} initialized")

    def _get_orchestrator(self, problem_name, problem_kwargs, model_params=None):
        """
        Return a cached orchestrator for the requested problem.

        Args:
            problem_name: Logical problem key, e.g. ``local_problem`` or
                ``aggregation_problem``.
            problem_kwargs: Full problem definition used when the orchestrator is
                first created.
            model_params: Optional state dict loaded into the orchestrator model.

        Returns:
            An ``OrchestratorWrapper`` ready to train or evaluate the problem.
        """

        if problem_name not in self.orchestrator_map:
            debug_print(f"Initializing orchestrator for {problem_name}")
            orchestrator = self.orchestrator_fn(logger=None, **problem_kwargs)
            assert isinstance(
                orchestrator, OrchestratorWrapper), f"Invalid orchestrator for {problem_name}"
            self.orchestrator_map[problem_name] = orchestrator
        else:
            debug_print(f"Using cached orchestrator for {problem_name}")

        orchestrator = self.orchestrator_map[problem_name]
        if model_params is not None:
            orchestrator.set_model_params(model_params)

        return orchestrator

    def setup(self, **kwargs):
        """
        Prepare the client actor before training starts.

        Args:
            **kwargs: Optional setup payload from the server. Currently accepted
                for interface compatibility.
        """
        debug_print("Setting up client")

    @profile
    def fit(self, **kwargs):
        """
        Run local optimization for a server-provided problem.

        Args:
            **kwargs: Expected keys are ``model_params``, ``problem``,
                ``num_local_epochs`` and ``num_global_epochs``. In the local
                FairLAB phase, ``num_global_epochs`` means orchestrator epochs
                and ``num_local_epochs`` means learner epochs. During
                aggregation, ``num_local_epochs`` controls the client learner
                epochs used for distillation, while ``aggregation_teacher_logits``
                may contain the global ensemble target.

        Returns:
            Dict with ``params`` containing the updated model state dict and
            ``weight`` containing this client's FedAvg contribution weight.
        """
        num_local_epochs = kwargs.get(
            'num_local_epochs', self.num_local_epochs)
        num_global_epochs = kwargs.get(
            'num_global_epochs', self.num_global_epochs)
        debug_print(
            f"Client {self.client_name} fitting for {num_global_epochs} orchestrator epochs and {num_local_epochs} learner epochs")
        problem = kwargs.get('problem')
        problem_name = problem['name']
        aggregation_teachers_list = problem.get(
            'aggregation_teachers_list', [])
        model_params = kwargs.get('model_params')

        aggregation_weights = problem.get('aggregation_weights', None)
        self.orchestrator = self._get_orchestrator(
            problem_name, problem, model_params)

        current_state = self.state if problem_name == 'local_problem' else None
        updated_model, state = self.orchestrator.fit(
            model_params=model_params,
            num_global_iterations=num_global_epochs,
            num_local_epochs=num_local_epochs,
            state=current_state,
            aggregation_teachers_list=aggregation_teachers_list,
            aggregation_weights=aggregation_weights,
            aggregation_teacher_logits=kwargs.get('aggregation_teacher_logits', None),)

        if problem_name == 'local_problem':
            self.state = copy.deepcopy(state)
            if state.get('performance_reference') is not None:
                self.performance_reference = float(
                    state['performance_reference'])
        result = {
            'params': _state_dict_to_cpu(updated_model.state_dict()),
            'weight': 1.0,
        }
        return result

    def compute_weighted_ensemble_logits(self, **kwargs):
        """
        Compute this client's contribution to the ensemble logits.

        Each candidate model is evaluated on the client's local data and combined
        using the server-provided ensemble weights. Raw data never leaves the
        client; only the weighted logits are returned.

        Args:
            **kwargs: Requires ``problem``, ``model_params_list`` and
                ``aggregation_weights``. ``use_training`` selects train or
                validation data for the logit query.

        Returns:
            Tensor of shape ``[num_local_examples, num_classes]``.
        """
        problem = kwargs.get('problem')
        assert problem is not None, "Problem is required"
        problem_name = problem['name']
        model_params_list = kwargs.get('model_params_list')
        aggregation_weights = kwargs.get('aggregation_weights')
        assert model_params_list is not None, "Model parameters list is required"
        assert aggregation_weights is not None, "Aggregation weights are required"

        self.orchestrator = self._get_orchestrator(problem_name, problem)
        return self.orchestrator.compute_weighted_ensemble_logits(
            model_params_list=model_params_list,
            weights=aggregation_weights,
            use_training=kwargs.get('use_training', True),
        )

    def evaluate_precomputed_ensemble_logits(self, **kwargs):
        """Evaluate an ensemble-logit target with the standard local metrics."""
        problem = kwargs['problem']
        orchestrator = self._get_orchestrator(problem['name'], problem)
        learner = orchestrator.main_problem.eval_subproblem.instance
        metrics = learner.evaluate_precomputed_logits(
            kwargs['ensemble_logits'], split='val')
        return {
            key: float(value)
            for key, value in metrics.items()
        }

    def update(self, **kwargs):
        """
        Placeholder for the base client update interface.

        FedFairLab performs updates through ``fit`` because each update depends
        on a full constrained problem definition.
        """
        pass

    def save(self, metrics, path):
        """
        Persist the client's selected local model and metrics.

        Args:
            metrics: Metric dictionary associated with the saved checkpoint.
            path: Destination checkpoint path.
        """
        save_dict = {
            'model_state_dict': self.model,
            'metrics': metrics}
        torch.save(save_dict, path)

    def load(self, path):
        """
        Load a client checkpoint.

        Args:
            path: Path to a checkpoint created by ``save``.

        Returns:
            Deserialized checkpoint dictionary.
        """
        return load_trusted_checkpoint(path)

    def _eval_and_log(self, **kwargs):
        """
        Select and log the best local evaluation among candidate models.

        Args:
            **kwargs: Contains global scoring inputs, candidate model parameters,
                and their local evaluation results.
        """
        performance_constraint = kwargs.get('performance_constraint')
        performance_reference = kwargs.get(
            'performance_reference', self.performance_reference)
        original_threshold_list = kwargs.get('original_threshold_list')
        self.local_scoring_config = {
            'performance_constraint': performance_constraint,
            'performance_reference': performance_reference,
            'original_threshold_list': tuple(original_threshold_list or ()),
            'first_performance_constraint': kwargs.get(
                'first_performance_constraint', False),
        }
        model_params_list = kwargs.get('model_params_list')
        eval_results = kwargs.get('eval_results')
        best_results = {}
        best_score = -np.inf
        best_model_params = None
        for model, result in zip(model_params_list, eval_results):
            local_result = compute_global_score(
                performance_constraint=performance_constraint,
                performance_reference=performance_reference,
                original_threshold_list=original_threshold_list,
                eval_results=[result],
            )
            local_result['metrics']['val_constraints_score'] = local_result['metrics']['val_global_score']
            del local_result['metrics']['val_global_score']
            if local_result['metrics']['val_constraints_score'] > best_score:
                best_score = local_result['metrics']['val_constraints_score']
                best_results = copy.deepcopy(local_result)
                best_model_params = model
        self.model = copy.deepcopy(best_model_params)
        metrics = best_results['metrics']
        for checkpoint in self.client_checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=partial(
                    self.save, metrics), metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        self.logger.log(metrics)
        return

    def evaluate(self, **kwargs):
        """
        Evaluate one model on this client's local data.

        Args:
            **kwargs: Requires ``problem`` and ``model_params``.

        Returns:
            Metric dictionary produced by the orchestrator.
        """
        problem = kwargs['problem']
        problem_name = problem['name']
        self.orchestrator = self._get_orchestrator(problem_name, problem)

        model_params = kwargs.get('model_params')
        assert model_params is not None, "Model parameters are required"
        results = self.orchestrator.evaluate(
            model_params, split=kwargs.get('split', 'val'))

        return results

    def evaluate_aggregation_distillation(self, **kwargs):
        """Evaluate one global model against this client's ensemble target.

        This RPC is used only for checkpoint selection inside the federated
        aggregation phase.  It evaluates the same validation distillation
        objective optimized locally and does not update FairLAB state.
        """
        problem = kwargs['problem']
        model_params = kwargs['model_params']
        ensemble_logits = kwargs['aggregation_teacher_logits']
        orchestrator = self._get_orchestrator(problem['name'], problem)
        orchestrator.main_problem.aggregation_teacher_logits = ensemble_logits
        orchestrator.main_problem.query_teachers()
        try:
            metrics = orchestrator.evaluate(model_params, split='val')
        finally:
            orchestrator.main_problem.aggregation_teacher_logits = None
        return {
            'val_distillation_loss': float(metrics['val_loss']),
            # The current cross-silo protocol and FedAvg implementation assign
            # equal mass to every participating client.
            'weight': 1.0,
        }

    def evaluate_constraints(self, **kwargs):
        """
        Evaluate global constraints for one model on this client.

        Args:
            **kwargs: Includes ``problem``, ``model_params``, scoring thresholds,
                and logging flags.

        Returns:
            Dictionary with local constraint values and task metrics in the
            format expected by server-side global scoring.
        """
        problem = kwargs['problem']
        problem_name = problem['name']
        self.orchestrator = self._get_orchestrator(problem_name, problem)

        model_params = kwargs.get('model_params')
        log_results = kwargs.get('log_results', True)
        performance_constraint = kwargs.get('performance_constraint')
        performance_reference = kwargs.get(
            'performance_reference', self.performance_reference)
        original_threshold_list = kwargs.get('original_threshold_list')
        problem = kwargs.get('problem')
        problem_name = problem['name']

        first_performance_constraint = kwargs.get(
            'first_performance_constraint', False)
        split = kwargs.get('split', 'val')
        if problem_name == 'global_problem':
            self.local_scoring_config = {
                'performance_constraint': performance_constraint,
                'performance_reference': performance_reference,
                'original_threshold_list': tuple(
                    original_threshold_list or ()),
                'first_performance_constraint': first_performance_constraint,
            }
        results_dict = self.orchestrator.evaluate_constraints2(
            model_params, split=split)
        constraints_key = f'{split}_constraints'
        final_results = {'train_constraints': [], constraints_key: []}

        for v in [constraints_key]:
            for key, value in results_dict[v]['macro_constraints_violations'].items():
                if first_performance_constraint and key == 0:
                    continue
                final_results[v].append(value[0])

        for v in results_dict.keys():
            if v not in [constraints_key]:
                final_results[v] = results_dict[v]
        if problem_name == 'global_problem' and split == 'val':
            if log_results:
                self._eval_and_log(
                    performance_constraint=performance_constraint,
                    performance_reference=performance_reference,
                    original_threshold_list=original_threshold_list,
                    first_performance_constraint=first_performance_constraint,
                    model_params_list=[model_params],
                    eval_results=[final_results],
                )
        return final_results

    def _get_final_results(self, **kwargs):
        """
        Load the best local checkpoint selected during training.

        Returns:
            Tuple ``(model_params, metrics, checkpoint_path)``.
        """
        checkpoint = self.client_checkpoints[0]
        assert isinstance(
            checkpoint, ModelCheckpoint), "Checkpoint must be an instance of ModelCheckpoint"
        best_results = self.load(checkpoint.get_model_path())
        file_path = checkpoint.get_model_path()
        best_metrics = best_results['metrics']
        best_model_params = best_results['model_state_dict']
        return best_model_params, best_metrics, file_path

    def _log_final_results(self, **kwargs):
        """
        Log final local metrics and upload the saved local model artifact.
        """
        model_params, validation_metrics, path = self._get_final_results(**kwargs)
        final_split = kwargs.get('final_split', 'test')
        if final_split == 'val':
            metric_prefix = kwargs.get('metric_prefix', 'final')
            final_results = {
                f'{metric_prefix}_local_{key}': value
                for key, value in validation_metrics.items()
            }
            self.logger.log(final_results)
            self.logger.log_artifact(
                f'{self.client_name}_local_model', path)
            return
        if final_split != 'test':
            raise ValueError(f"Unsupported final split: {final_split}")
        orchestrator = self.orchestrator_map.get('global_problem')
        if orchestrator is None:
            raise RuntimeError(
                "Global evaluation problem is unavailable for local test evaluation")

        scoring_config = self.local_scoring_config or {
            'performance_constraint': None,
            'performance_reference': self.performance_reference,
            'original_threshold_list': (),
            'first_performance_constraint': False,
        }
        results_dict = orchestrator.evaluate_constraints2(
            model_params, split='test')
        test_constraints = []
        for key, value in results_dict['test_constraints'][
                'macro_constraints_violations'].items():
            if (scoring_config['first_performance_constraint'] and key == 0):
                continue
            test_constraints.append(value[0])

        local_test_result = {
            'test_constraints': test_constraints,
            'test_objective_fn': results_dict['test_objective_fn'],
            'metrics': results_dict['metrics'],
        }
        scored_test_result = compute_global_score(
            performance_constraint=scoring_config['performance_constraint'],
            performance_reference=scoring_config.get(
                'performance_reference'),
            original_threshold_list=scoring_config['original_threshold_list'],
            eval_results=[local_test_result],
            split='test',
        )
        test_metrics = scored_test_result['metrics']
        test_metrics['test_constraints_score'] = test_metrics.pop(
            'test_global_score')
        final_results = _prefix_final_local_metrics(
            validation_metrics,
            test_metrics,
        )
        self.logger.log(final_results)
        self.logger.log_artifact(f'{self.client_name}_local_model', path)

    @profile
    def evaluate_constraints_list(self, **kwargs):
        """
        Evaluate multiple candidate models on this client's local data.

        Args:
            **kwargs: Requires ``problem`` and ``model_params_list`` plus the
                global scoring configuration.

        Returns:
            List of local evaluation dictionaries, one per candidate model.
        """
        problem = kwargs['problem']
        problem_name = problem['name']
        self.orchestrator = self._get_orchestrator(problem_name, problem)

        log_results = kwargs.get('log_results', True)
        problem = kwargs.get('problem')
        problem_name = problem['name']
        model_params_list = kwargs.get('model_params_list')
        first_performance_constraint = kwargs.get(
            'first_performance_constraint', False)

        performance_constraint = kwargs.get('performance_constraint')
        performance_reference = kwargs.get(
            'performance_reference', self.performance_reference)
        original_threshold_list = kwargs.get('original_threshold_list')

        results = []

        for model_params in model_params_list:
            results_dict = self.orchestrator.evaluate_constraints2(
                model_params)

            final_results = {'train_constraints': [], 'val_constraints': []}

            for v in ['val_constraints']:
                for key, value in results_dict[v]['macro_constraints_violations'].items():
                    if first_performance_constraint and key == 0:
                        continue
                    final_results[v].append(value[0])

            for v in results_dict.keys():
                if v not in ['val_constraints']:
                    final_results[v] = results_dict[v]

            results.append(final_results)
        if problem_name == 'global_problem':
            if log_results:
                self._eval_and_log(
                    performance_constraint=performance_constraint,
                    performance_reference=performance_reference,
                    original_threshold_list=original_threshold_list,
                    first_performance_constraint=first_performance_constraint,
                    model_params_list=model_params_list,
                    eval_results=results,
                )

        return results

    def shutdown(self, **kwargs):
        """
        Close the client logger and optionally log final local results.

        Args:
            **kwargs: ``log_results`` controls whether final checkpoint metrics
                and artifacts are emitted.
        """
        log_results = kwargs.get('log_results', True)
        if log_results:
            self._log_final_results(**kwargs)
        self.logger.close()

    def fine_tune(self, **kwargs):
        """
        Delegate optional fine-tuning to the base implementation.

        Args:
            **kwargs: Fine-tuning payload.
        """
        return super().fine_tune(**kwargs)
