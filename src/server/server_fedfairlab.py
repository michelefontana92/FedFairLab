from debug_utils import debug_print
from .server_base import BaseServer
import ray

from .server_factory import register_server
from callbacks.early_stopping import EarlyStopping
from callbacks.model_checkpoint import ModelCheckpoint
from loggers.wandb_logger import WandbLogger
from functools import partial
import os
import torch
import copy
from surrogates import SurrogateFactory
from .utils import compute_group_cardinality, compute_global_score
from tqdm import tqdm
import random
import math
from checkpoint_utils import load_trusted_checkpoint


DISTILLATION_TASK_WEIGHT = 0.8
DISTILLATION_TEMPERATURE = 2.0
ENSEMBLE_WEIGHT_TEMPERATURE = 0.05
TEACHER_SAMPLING_TEMPERATURE = 0.05


class EarlyStoppingException(Exception):
    """Raised internally to stop server training when early stopping triggers."""
    pass


def _resolve_server_patience(options):
    """Return the patience used to select the global validation checkpoint."""
    return int(options.get('global_patience', 10))


@register_server("server_fedfairlab")
class ServerFedFairLab(BaseServer):
    """
    FedFairLab server coordinating client updates and global aggregation.

    The server implements the global phase of FedFairLab: teacher sampling from a
    bounded history, local client mitigation, client-aligned ensemble-logit
    construction, federated distillation via FedAvg, global scoring, and
    checkpointing.
    """

    def __init__(self, **kwargs):
        """
        Initialize the server, scoring problems, callbacks, and global history.

        Args:
            **kwargs: Experiment configuration including model, client builders,
                fairness metrics, sensitive groups, thresholds, performance
                budget, server patience, federation size, history size,
                aggregation/distillation rounds, and local client epochs used
                during aggregation distillation.
        """

        self.clients_init_fn_list = kwargs.get('clients_init_fn_list')
        self.model = kwargs.get('model')
        self.log_model = kwargs.get('log_model', False)
        self.project = kwargs.get('project_name', 'fedfairlab')
        self.id = kwargs.get('server_name', 'server')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_name', 'global_model.h5')
        self.patience = _resolve_server_patience(kwargs)
        self.verbose = kwargs.get('verbose', False)
        self.num_federated_iterations = kwargs.get(
            'num_federated_iterations', 1)
        self.aggregation_epochs = max(1, kwargs.get('aggregation_epochs', 1))
        self.aggregation_local_epochs = max(
            1, kwargs.get('aggregation_local_epochs', 10))
        self.aggregation_patience = max(
            0, int(kwargs.get('aggregation_patience', 0)))
        self.aggregation_min_delta = max(
            0.0, float(kwargs.get('aggregation_min_delta', 1e-6)))
        self.history_size = max(1, kwargs.get('history_size', 5))
        self._last_teacher_sampling_diagnostics = {}
        self.num_classes = kwargs.get('num_classes', 2)

        self.original_metrics_list = kwargs.get('metrics_list')
        self.original_groups_list = kwargs.get('groups_list')
        self.original_threshold_list = kwargs.get('threshold_list')
        self.original_metrics_list = tuple(self.original_metrics_list or ())
        self.original_groups_list = tuple(self.original_groups_list or ())
        self.original_threshold_list = tuple(
            self.original_threshold_list or ())
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.performance_constraint = kwargs.get('performance_constraint')
        self.performance_step = kwargs.get('performance_step', 0.0)
        self.global_performance_reference = None
        self.history_global = []
        self.fraction = float(kwargs.get('fraction', 0.5))
        if not 0 < self.fraction <= 1:
            raise ValueError("Client fraction must be in the interval (0, 1].")

        self.callbacks = [
            EarlyStopping(patience=self.patience,
                          monitor='val_global_score',
                          mode='max'
                          ),
            ModelCheckpoint(save_dir=self.checkpoint_dir,
                            save_name=self.checkpoint_name,
                            monitor='val_global_score',
                            mode='max')
        ]

        self.logger = WandbLogger(
            project=self.project,
            config=None,
            id=self.id,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_path=self.checkpoint_name,
            log_model=self.log_model,
            data_module=self.data if self.log_model else None
        )

        self.problem = self._init_constrained_problem(
            use_adaptive_aggregation=True, **kwargs)
        self.global_problem = self._init_global_constrained_problem(**kwargs)
        self.aggregation_problem = self._init_aggregation_problem(**kwargs)

        self.global_model = None
        debug_print('Server initialized')

    def aggregation_phase(self, **kwargs):
        """Distill the weighted local ensemble and return its best FedAvg model."""
        aggregation_epochs = kwargs.get(
            'aggregation_epochs', self.aggregation_epochs)
        num_local_epochs = kwargs.get(
            'num_local_epochs', self.aggregation_local_epochs)
        params = kwargs.get('params')
        model_params_list = [p['params'] for p in params]
        selected_clients = kwargs.get('selected_clients', self.clients)
        teacher_model_params = kwargs.get(
            'teacher_model_params', self.model.state_dict())
        assert len(model_params_list) > 0, "Model parameters are required"

        scores, model_eval_list = self.evaluate_list(
            model_params_list=model_params_list,
            return_metrics=True,
            log_results=True
        )
        ensemble_weights, weight_diagnostics = (
            self._compute_ensemble_weights(scores)
        )
        candidate_diagnostics = self._candidate_aggregation_diagnostics(
            scores=scores,
            weights=ensemble_weights,
            model_eval_list=model_eval_list,
        )

        distillation_problem = copy.deepcopy(self.aggregation_problem)
        distillation_problem['aggregation_teachers_list'] = []
        for key in ['objective_function', 'original_objective_function', 'batch_objective_function']:
            distillation_problem[key].set_weights([1.0])
        distillation_problem['aggregation_weights'] = [1.0]

        student_params = copy.deepcopy(teacher_model_params)

        client_train_logits = self.compute_client_ensemble_logits(
            model_params_list=model_params_list,
            aggregation_weights=ensemble_weights,
            selected_clients=selected_clients,
            use_training=True,
        )
        client_val_logits = self.compute_client_ensemble_logits(
            model_params_list=model_params_list,
            aggregation_weights=ensemble_weights,
            selected_clients=selected_clients,
            use_training=False,
        )
        ensemble_diagnostics = self._evaluate_ensemble_logits_metrics(
            client_val_logits=client_val_logits,
            selected_clients=selected_clients,
        )
        if not (
            len(client_train_logits)
            == len(client_val_logits)
            == len(selected_clients)
        ):
            raise ValueError(
                "Expected one train and validation ensemble target per client")

        best_student_params = copy.deepcopy(student_params)
        best_validation_kd_loss = self._evaluate_federated_distillation_loss(
            model_params=student_params,
            problem=distillation_problem,
            client_val_logits=client_val_logits,
            selected_clients=selected_clients,
        )
        best_distillation_epoch = 0
        last_validation_kd_loss = best_validation_kd_loss
        epochs_without_improvement = 0
        aggregation_epochs_executed = 0
        aggregation_early_stopped = 0

        for aggregation_epoch in range(1, aggregation_epochs + 1):
            aggregation_epochs_executed = aggregation_epoch
            client_jobs = [
                client.fit.remote(
                    model_params=student_params,
                    problem=distillation_problem,
                    num_local_epochs=num_local_epochs,
                    num_global_epochs=1,
                    aggregation_teacher_logits={
                        'train': copy.deepcopy(train_logits),
                        'val': copy.deepcopy(val_logits),
                    },
                )
                for client, train_logits, val_logits in zip(
                    selected_clients, client_train_logits, client_val_logits)
            ]
            client_results = ray.get(client_jobs)
            student_params = self.fedavg_model_params(client_results)
            last_validation_kd_loss = (
                self._evaluate_federated_distillation_loss(
                    model_params=student_params,
                    problem=distillation_problem,
                    client_val_logits=client_val_logits,
                    selected_clients=selected_clients,
                )
            )
            min_delta = float(getattr(
                self, 'aggregation_min_delta', 1e-6))
            if last_validation_kd_loss < best_validation_kd_loss - min_delta:
                best_validation_kd_loss = last_validation_kd_loss
                best_distillation_epoch = aggregation_epoch
                best_student_params = copy.deepcopy(student_params)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                patience = int(getattr(self, 'aggregation_patience', 0))
                if (
                    patience > 0
                    and epochs_without_improvement >= patience
                ):
                    aggregation_early_stopped = 1
                    break

        student_params = best_student_params

        client_evaluations = self._broadcast_fn(
            'evaluate_constraints',
            model_params=student_params,
            problem=self.global_problem,
            first_performance_constraint=self._has_performance_budget(),
            performance_constraint=self.performance_constraint,
            original_threshold_list=self.original_threshold_list,
            log_results=True
        )

        final_model_eval = self._compute_global_score(
            eval_results=client_evaluations,
            update_performance_reference=True,
        )
        final_model_eval['metrics'].update(weight_diagnostics)
        final_model_eval['metrics'].update(candidate_diagnostics)
        final_model_eval['metrics'].update(ensemble_diagnostics)
        final_model_eval['metrics'].update({
            'aggregation_best_distillation_loss': best_validation_kd_loss,
            'aggregation_best_distillation_epoch': best_distillation_epoch,
            'aggregation_last_distillation_loss': last_validation_kd_loss,
            'aggregation_epochs_executed': aggregation_epochs_executed,
            'aggregation_early_stopped': aggregation_early_stopped,
            'aggregation_rounds_without_improvement': (
                epochs_without_improvement
            ),
        })
        for metric_name, metric_value in list(
                final_model_eval['metrics'].items()):
            if metric_name == 'val_f1' or metric_name.startswith(
                    'val_demographic_parity_'):
                final_model_eval['metrics'][
                    f'aggregation_student_{metric_name}'] = float(metric_value)
        return student_params, final_model_eval

    def _candidate_aggregation_diagnostics(
            self, *, scores, weights, model_eval_list):
        """Return flat logging metrics for local ensemble candidates."""
        diagnostics = {}
        for idx, (score, weight, evaluation) in enumerate(zip(
                scores, weights, model_eval_list)):
            prefix = f'aggregation_candidate_{idx}'
            diagnostics[f'{prefix}_score'] = float(score)
            diagnostics[f'{prefix}_weight'] = float(weight)
            metrics = evaluation.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                if metric_name == 'val_f1' or metric_name.startswith(
                        'val_demographic_parity_'):
                    diagnostics[f'{prefix}_{metric_name}'] = float(metric_value)
        return diagnostics

    def _evaluate_ensemble_logits_metrics(
            self, *, client_val_logits, selected_clients):
        """Evaluate the ensemble targets on client validation splits."""
        handlers = [
            client.evaluate_precomputed_ensemble_logits.remote(
                problem=self.global_problem,
                ensemble_logits=copy.deepcopy(logits),
            )
            for client, logits in zip(selected_clients, client_val_logits)
        ]
        local_metrics = ray.get(handlers)
        if not local_metrics:
            return {}
        common_keys = set(local_metrics[0])
        for metrics in local_metrics[1:]:
            common_keys.intersection_update(metrics)
        return {
            f'aggregation_ensemble_{metric_name}': sum(
                float(metrics[metric_name]) for metrics in local_metrics
            ) / len(local_metrics)
            for metric_name in sorted(common_keys)
            if metric_name == 'val_f1' or metric_name.startswith(
                'val_demographic_parity_')
        }

    def _evaluate_federated_distillation_loss(
            self, *, model_params, problem, client_val_logits,
            selected_clients):
        """Return the weighted mean validation KD loss across clients."""
        if len(client_val_logits) != len(selected_clients):
            raise ValueError(
                'Expected one validation ensemble target per selected client')
        handlers = [
            client.evaluate_aggregation_distillation.remote(
                model_params=copy.deepcopy(model_params),
                problem=problem,
                aggregation_teacher_logits={
                    'val': copy.deepcopy(val_logits),
                },
            )
            for client, val_logits in zip(
                selected_clients, client_val_logits)
        ]
        evaluations = ray.get(handlers)
        total_weight = sum(float(result.get('weight', 1.0))
                           for result in evaluations)
        if total_weight <= 0.0:
            raise ValueError(
                'Federated distillation evaluation requires positive weight')
        return sum(
            float(result['val_distillation_loss'])
            * float(result.get('weight', 1.0))
            for result in evaluations
        ) / total_weight

    def _compute_ensemble_weights(self, scores):
        """Convert candidate global scores into normalized ensemble weights."""
        if not scores:
            raise ValueError('At least one candidate score is required')

        score_tensor = torch.tensor(scores, dtype=torch.float32)
        weight_tensor = torch.softmax(
            score_tensor / ENSEMBLE_WEIGHT_TEMPERATURE, dim=0)
        entropy = -torch.sum(
            weight_tensor * torch.log(weight_tensor.clamp_min(1e-12)))
        effective_teachers = torch.exp(entropy)
        if weight_tensor.numel() > 1:
            normalized_entropy = entropy / torch.log(torch.tensor(
                float(weight_tensor.numel()), dtype=entropy.dtype))
        else:
            normalized_entropy = torch.ones_like(entropy)

        diagnostics = {
            'aggregation_weight_temperature': ENSEMBLE_WEIGHT_TEMPERATURE,
            'aggregation_weight_max': float(weight_tensor.max().item()),
            'aggregation_weight_min': float(weight_tensor.min().item()),
            'aggregation_weight_entropy': float(entropy.item()),
            'aggregation_weight_normalized_entropy': float(
                normalized_entropy.item()),
            'aggregation_effective_teachers': float(
                effective_teachers.item()),
            'aggregation_score_min': float(score_tensor.min().item()),
            'aggregation_score_max': float(score_tensor.max().item()),
            'aggregation_score_spread': float(
                (score_tensor.max() - score_tensor.min()).item()),
        }
        return weight_tensor.tolist(), diagnostics

    def fedavg_model_params(self, client_results):
        """
        Average client model parameters with FedAvg.

        Args:
            client_results: List of dictionaries with ``params`` state dicts and
                scalar ``weight`` values.

        Returns:
            State dict containing the weighted average of client parameters.
        """
        total_weight = sum(result['weight'] for result in client_results)
        assert total_weight > 0, "FedAvg requires positive client weights"

        averaged_params = {}
        for key in client_results[0]['params']:
            averaged_params[key] = sum(
                result['params'][key] * (result['weight'] / total_weight)
                for result in client_results
            )
        return averaged_params

    def _has_performance_budget(self):
        """
        Check whether the FairLAB-style performance budget is active.

        Returns:
            True when beta was provided and should be interpreted as the
            tolerated drop from the best global F1 observed so far.
        """
        return self.performance_constraint is not None

    def _compute_global_score(self, eval_results, update_performance_reference=False,
                              split='val'):
        """
        Compute the server global score with FairLAB budget semantics.

        The global performance penalty uses the same interpretation as the local
        FairLAB constraints: ``F1_global >= p*_global - beta``. When requested,
        ``p*_global`` is updated from the evaluated model before the final score is
        returned.

        Args:
            eval_results: Per-client evaluation dictionaries.
            update_performance_reference: If true, refresh ``p*_global`` from the
                aggregated validation F1 before scoring.
            split: Evaluation split. Test evaluation never updates the reference.

        Returns:
            Global score dictionary.
        """
        score_dict = compute_global_score(
            performance_constraint=self.performance_constraint,
            performance_reference=self.global_performance_reference,
            original_threshold_list=self.original_threshold_list,
            eval_results=eval_results,
            constraint_weight=1.0,
            split=split,
        )

        if split != 'val' and update_performance_reference:
            raise ValueError(
                "The performance reference can only be updated on validation data")

        if (
            update_performance_reference
            and self._has_performance_budget()
        ):
            current_performance = float(score_dict['val_objective_fn'])
            if self.global_performance_reference is None:
                self.global_performance_reference = current_performance
            else:
                self.global_performance_reference = max(
                    self.global_performance_reference,
                    current_performance,
                )
            score_dict = compute_global_score(
                performance_constraint=self.performance_constraint,
                performance_reference=self.global_performance_reference,
                original_threshold_list=self.original_threshold_list,
                eval_results=eval_results,
                constraint_weight=1.0,
                split=split,
            )

        return score_dict

    def update_global_history(self, model_params, eval_result):
        """
        Insert a global model into the bounded server history.

        Args:
            model_params: Global model state dict to store.
            eval_result: Global evaluation dictionary containing
                ``metrics['val_global_score']``.
        """
        score = eval_result['metrics']['val_global_score']
        self.history_global.append({
            'params': copy.deepcopy(model_params),
            'score': score,
            'eval': copy.deepcopy(eval_result),
        })
        self.history_global.sort(key=lambda item: item['score'], reverse=True)
        self.history_global = self.history_global[:self.history_size]

    def sample_teacher_from_history(self):
        """
        Sample a global teacher model from history using Boltzmann probabilities.

        Returns:
            Tuple ``(model_params, selected_idx)``. If history is empty, returns
            the current server model and ``None``.
        """
        if len(self.history_global) == 0:
            self._last_teacher_sampling_diagnostics = {
                'history_teacher_selected_index': -1,
                'history_teacher_sampling_temperature': float(
                    TEACHER_SAMPLING_TEMPERATURE),
                'history_teacher_pool_size': 0,
            }
            return copy.deepcopy(self.model.state_dict()), None

        scores = torch.tensor(
            [item['score'] for item in self.history_global],
            dtype=torch.float32,
        )
        probabilities = torch.softmax(
            scores / TEACHER_SAMPLING_TEMPERATURE, dim=0)
        selected_idx = torch.multinomial(probabilities, num_samples=1).item()
        selected = self.history_global[selected_idx]
        diagnostics = {
            'history_teacher_selected_index': selected_idx,
            'history_teacher_selected_score': float(selected['score']),
            'history_teacher_selected_probability': float(
                probabilities[selected_idx]),
            'history_teacher_sampling_temperature': float(
                TEACHER_SAMPLING_TEMPERATURE),
            'history_teacher_pool_size': len(self.history_global),
        }
        selected_metrics = selected.get('eval', {}).get('metrics', {})
        if 'val_f1' in selected_metrics:
            diagnostics['history_teacher_selected_val_f1'] = float(
                selected_metrics['val_f1'])
        for metric_name, metric_value in selected_metrics.items():
            if metric_name.startswith('val_demographic_parity_'):
                diagnostics[
                    f'history_teacher_selected_{metric_name}'
                ] = float(metric_value)
        for idx, (item, probability) in enumerate(zip(
                self.history_global, probabilities)):
            prefix = f'history_teacher_candidate_{idx}'
            diagnostics[f'{prefix}_score'] = float(item['score'])
            diagnostics[f'{prefix}_probability'] = float(probability)
            metrics = item.get('eval', {}).get('metrics', {})
            if 'val_f1' in metrics:
                diagnostics[f'{prefix}_val_f1'] = float(metrics['val_f1'])
            for metric_name, metric_value in metrics.items():
                if metric_name.startswith('val_demographic_parity_'):
                    diagnostics[f'{prefix}_{metric_name}'] = float(
                        metric_value)
        self._last_teacher_sampling_diagnostics = diagnostics
        # print(
        #    f"[FedFairLab] Selected teacher {selected_idx} "
        #    f"from history with score {selected['score']:.4f}"
        # )
        return copy.deepcopy(selected['params']), selected_idx

    def compute_client_ensemble_logits(
            self, model_params_list, aggregation_weights, selected_clients,
            use_training=True):
        """
        Ask clients to compute weighted candidate logits on their own examples.

        Args:
            model_params_list: Candidate local models returned by selected clients.
            aggregation_weights: Softmax-normalized global-score weights.
            selected_clients: Ray actors participating in this aggregation round.
            use_training: If true, query train logits; otherwise validation logits.

        Returns:
            List containing one client-aligned distillation target per client.

        Row indices only have meaning within a client's private dataset. The
        returned tensors are therefore kept separate and must never be averaged
        element-wise across clients.
        """
        handlers = [
            client.compute_weighted_ensemble_logits.remote(
                model_params_list=model_params_list,
                aggregation_weights=aggregation_weights,
                problem=self.aggregation_problem,
                use_training=use_training,
            )
            for client in selected_clients
        ]
        return ray.get(handlers)

    def evaluate_list(self, model_params_list, *, return_metrics=False, log_results=False):
        """
        Evaluate candidate models across all clients and compute global scores.

        Args:
            model_params_list: List of model state dicts to evaluate.
            return_metrics: If true, also return the full global metric dicts.
            log_results: If true, allow clients to log local evaluation results.

        Returns:
            List of global scores, or ``(scores, metric_dicts)`` when
            ``return_metrics`` is true.
        """
        assert len(
            model_params_list) > 0, "model_params_list non può essere vuota"

        eval_results = self._broadcast_fn(
            'evaluate_constraints_list',
            model_params_list=model_params_list,
            problem=self.global_problem,
            first_performance_constraint=self._has_performance_budget(),
            performance_constraint=self.performance_constraint,
            original_threshold_list=self.original_threshold_list,
            log_results=log_results
        )

        zipped_results = list(zip(*eval_results))
        global_scores = []
        model_eval_list = []

        for clientwise_eval in zipped_results:
            score_dict = self._compute_global_score(
                eval_results=clientwise_eval,
                update_performance_reference=False,
            )
            global_scores.append(score_dict['metrics']['val_global_score'])
            model_eval_list.append(score_dict)

        if return_metrics:
            return global_scores, model_eval_list
        return global_scores

    def evaluate(self, **kwargs):
        """
        Evaluate one model globally or on a single client.

        Args:
            **kwargs: Requires ``model_params``. Optional ``client_id`` restricts
                evaluation to a single client.

        Returns:
            Global-score dictionary produced by ``compute_global_score``.
        """
        model_params = kwargs.get('model_params')
        client_id = kwargs.get('client_id')
        split = kwargs.get('split', 'val')
        if split not in {'val', 'test'}:
            raise ValueError(f"Unsupported evaluation split: {split}")

        assert model_params is not None, "Model parameters are required"
        if client_id is None:
            results = self._broadcast_fn('evaluate_constraints',
                                         model_params=copy.deepcopy(
                                             model_params),
                                         problem=self.global_problem,
                                         first_performance_constraint=self._has_performance_budget(),
                                         performance_constraint=self.performance_constraint,
                                         original_threshold_list=self.original_threshold_list,
                                         split=split,
                                         log_results=kwargs.get('log_results', split == 'val'))

        else:
            handler = self.clients[client_id].evaluate_constraints.remote(model_params=copy.deepcopy(model_params),
                                                                          problem=self.global_problem,
                                                                          first_performance_constraint=self._has_performance_budget(),
                                                                              performance_constraint=self.performance_constraint,
                                                                              original_threshold_list=self.original_threshold_list,
                                                                              split=split,
                                                                              log_results=kwargs.get('log_results', split == 'val'))

            results = [ray.get(handler)]
        # print('Length of global results:',len(results))
        global_scores = self._compute_global_score(
            eval_results=results,
            update_performance_reference=kwargs.get(
                'update_performance_reference', False),
            split=split,
        )
        return global_scores

    def _create_clients(self, clients_init_fn_list):
        """
        Instantiate all client actors from builder-provided callables.

        Args:
            clients_init_fn_list: List of partial functions creating clients.

        Returns:
            List of client actor handles.
        """
        client_list = [client_init_fn()
                       for client_init_fn in clients_init_fn_list]
        # print('Clients:',client_list)
        return client_list

    def _init_constrained_problem(self, **kwargs):
        """
        Build the client-side local constrained optimization problem.

        Args:
            **kwargs: Global experiment configuration, including fairness metrics,
                groups, thresholds, performance budget and number of classes.

        Returns:
            Problem dictionary consumed by client orchestrators.
        """
        use_adaptive_aggregation = kwargs.get(
            'use_adaptive_aggregation', False)
        performance_score_surrogate = (
            'multiclass_f1'
            if self.num_classes > 2
            else 'binary_f1'
        )
        if use_adaptive_aggregation:
            objective_function = SurrogateFactory.create(
                name='fedfairlab_local_adaptive_objective',
                surrogate_name='fedfairlab_local_adaptive',
                weight=1,
                average='weighted',
                xi=DISTILLATION_TASK_WEIGHT,
                temperature=DISTILLATION_TEMPERATURE)
            original_objective_function = SurrogateFactory.create(
                name=performance_score_surrogate,
                mode='max',
                surrogate_name=performance_score_surrogate,
                weight=1,
                average='weighted')
            batch_objective_function = SurrogateFactory.create(
                name='fedfairlab_local_adaptive_batch_objective',
                surrogate_name='fedfairlab_local_adaptive_batch',
                weight=1,
                average='weighted',
                xi=DISTILLATION_TASK_WEIGHT,
                temperature=DISTILLATION_TEMPERATURE)
        else:
            objective_function = SurrogateFactory.create(
                name='performance', surrogate_name='cross_entropy', weight=1, average='weighted')
            original_objective_function = SurrogateFactory.create(
                name=performance_score_surrogate,
                mode='max',
                surrogate_name=performance_score_surrogate,
                weight=1,
                average='weighted')
            batch_objective_function = SurrogateFactory.create(
                name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')

        inequality_constraints = []
        macro_constraints = []
        shared_macro_constraints = []
        idx_constraint = 0
        all_group_ids = {}
        if self._has_performance_budget():
            # Constraint 0 encourages improvement p >= p* + rho. Constraint 1
            # is the budget floor p >= p* - beta. Both are soft FairLAB
            # constraints and p* is initialized/updated from validation only.
            performance_surrogate = 'multiclass_f1' if self.num_classes > 2 else 'binary_f1'
            inequality_constraints = [
                SurrogateFactory.create(name=performance_surrogate,
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False),
                SurrogateFactory.create(name=performance_surrogate, 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False)
            ]
            idx_constraint = 2
            macro_constraints = [[0, 1]]
            shared_macro_constraints = [0]

        for metric, group, threshold in zip(self.original_metrics_list,
                                            self.original_groups_list,
                                            self.original_threshold_list):

            group_cardinality = compute_group_cardinality(
                group, sensitive_attributes=self.sensitive_attributes)
            macro_constraint = []
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)
            for i in range(group_cardinality):
                for j in range(i+1, group_cardinality):
                    if self.num_classes == 2:
                        constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                             surrogate_name=f'diff_{metric}_{group}',
                                                             surrogate_weight=1,
                                                             average='weighted',
                                                             group_name=group,
                                                             unique_group_ids={group: list(
                                                                 range(group_cardinality))},
                                                             lower_bound=threshold,
                                                             use_max=False,
                                                             target_groups=torch.tensor([i, j]))

                        inequality_constraints.append(constraint)
                        macro_constraint.append(idx_constraint)
                        idx_constraint += 1
                    else:
                        for c in range(self.num_classes):
                            constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                                 surrogate_name=f'diff_{metric}_{group}',
                                                                 surrogate_weight=1,
                                                                 average='weighted',
                                                                 group_name=group,
                                                                 unique_group_ids={group: list(
                                                                     range(group_cardinality))},
                                                                 lower_bound=threshold,
                                                                 use_max=False,
                                                                 target_groups=torch.tensor(
                                                                     [i, j]),
                                                                 target_class=c
                                                                 )
                            inequality_constraints.append(constraint)
                            macro_constraint.append(idx_constraint)
                            idx_constraint += 1
            macro_constraints.append(macro_constraint)

        inequality_constraints = inequality_constraints
        macro_constraints = macro_constraints
        shared_macro_constraints = shared_macro_constraints
        all_group_ids = all_group_ids

        problem = {
            'name': 'local_problem',
            'original_objective_function': original_objective_function,
            'objective_function': objective_function,
            'batch_objective_function': batch_objective_function,
            'inequality_constraints': inequality_constraints,
            'macro_constraints_list': macro_constraints,
            'shared_macro_constraints': shared_macro_constraints,
            'all_group_ids': all_group_ids,
            'aggregation_teachers_list': [],
            'num_classes': self.num_classes,
            'performance_constraint': self.performance_constraint,
            'performance_step': self.performance_step,
        }
        debug_print('All group ids: ', all_group_ids)
        debug_print('Macro constraints: ', macro_constraints)
        debug_print('Num of macro constraints: ', len(macro_constraints))
        # print('Inequality constraints: ', inequality_constraints)
        debug_print('Num of inequality constraints: ', len(inequality_constraints))
        return problem

    def _init_aggregation_problem(self, **kwargs):
        """
        Build the unconstrained problem used during global distillation.

        Args:
            **kwargs: Experiment configuration used to derive group metadata.

        Returns:
            Problem dictionary for the aggregation/distillation phase.
        """

        objective_function = SurrogateFactory.create(
            name='fedfairlab_ensemble_distillation',
            surrogate_name='fedfairlab_ensemble_distillation',
            weight=1,
            average='weighted',
            temperature=DISTILLATION_TEMPERATURE)
        original_objective_function = SurrogateFactory.create(
            name='fedfairlab_ensemble_distillation_score',
            surrogate_name='fedfairlab_ensemble_distillation_score',
            weight=1,
            average='weighted',
            temperature=DISTILLATION_TEMPERATURE)
        batch_objective_function = SurrogateFactory.create(
            name='fedfairlab_ensemble_distillation_batch',
            surrogate_name='fedfairlab_ensemble_distillation_batch',
            weight=1,
            average='weighted',
            temperature=DISTILLATION_TEMPERATURE)
        all_group_ids = {}
        for metric, group, threshold in zip(self.original_metrics_list,
                                            self.original_groups_list,
                                            self.original_threshold_list):

            group_cardinality = compute_group_cardinality(
                group, sensitive_attributes=self.sensitive_attributes)
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)

        problem = {
            'name': 'aggregation_problem',
            'original_objective_function': original_objective_function,
            'objective_function': objective_function,
            'batch_objective_function': batch_objective_function,
            'inequality_constraints': [],
            'macro_constraints_list': [],
            'shared_macro_constraints': [],
            'all_group_ids': all_group_ids,
            'aggregation_teachers_list': [],
            'num_classes': self.num_classes,
            'performance_constraint': self.performance_constraint,
            'performance_step': self.performance_step,

        }
        debug_print('Aggregation: All group ids: ', all_group_ids)
        return problem

    def _init_global_constrained_problem(self, **kwargs):
        """
        Build the server-side global evaluation problem.

        The problem mirrors the fairness and performance requirements used for
        global scoring, but is evaluated by broadcasting model parameters to all
        clients and aggregating their local metrics.

        Args:
            **kwargs: Experiment configuration with fairness requirements.

        Returns:
            Problem dictionary used for global evaluation.
        """
        objective_function = SurrogateFactory.create(
            name='performance', surrogate_name='cross_entropy', weight=1, average='weighted')
        batch_objective_function = SurrogateFactory.create(
            name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')
        performance_score_surrogate = (
            'multiclass_f1'
            if self.num_classes > 2
            else 'binary_f1'
        )
        original_objective_function = SurrogateFactory.create(
            name=performance_score_surrogate,
            mode='max',
            surrogate_name=performance_score_surrogate,
            weight=1,
            average='weighted')

        inequality_constraints = []
        macro_constraints = []
        shared_macro_constraints = []
        idx_constraint = 0
        all_group_ids = {}
        if self._has_performance_budget():
            performance_surrogate = 'multiclass_f1' if self.num_classes > 2 else 'binary_f1'
            inequality_constraints = [
                SurrogateFactory.create(name=performance_surrogate, 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False),
                SurrogateFactory.create(name=performance_surrogate, 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False)
            ]
            idx_constraint = 2
            macro_constraints = [[0, 1]]
            shared_macro_constraints = [0]

        for metric, group, _ in zip(self.original_metrics_list,
                                    self.original_groups_list,
                                    self.original_threshold_list):
            group_cardinality = compute_group_cardinality(
                group, sensitive_attributes=self.sensitive_attributes)
            macro_constraint = []
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)
            for i in range(group_cardinality):
                for j in range(i+1, group_cardinality):
                    if self.num_classes == 2:
                        constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                             surrogate_name=f'diff_{metric}_{group}',
                                                             surrogate_weight=1,
                                                             average='weighted',
                                                             group_name=group,
                                                             unique_group_ids={group: list(
                                                                 range(group_cardinality))},
                                                             lower_bound=0.0,
                                                             use_max=True,
                                                             target_groups=torch.tensor(
                                                                 [i, j])
                                                             )
                        inequality_constraints.append(constraint)
                        macro_constraint.append(idx_constraint)
                        idx_constraint += 1
                    else:
                        for c in range(self.num_classes):
                            constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                                 surrogate_name=f'diff_{metric}_{group}',
                                                                 surrogate_weight=1,
                                                                 average='weighted',
                                                                 group_name=group,
                                                                 unique_group_ids={group: list(
                                                                     range(group_cardinality))},
                                                                 lower_bound=0.0,
                                                                 use_max=True,
                                                                 target_groups=torch.tensor(
                                                                     [i, j]),
                                                                 target_class=c
                                                                 )
                            inequality_constraints.append(constraint)
                            macro_constraint.append(idx_constraint)
                            idx_constraint += 1
            macro_constraints.append(macro_constraint)

        inequality_constraints = inequality_constraints
        macro_constraints = macro_constraints
        shared_macro_constraints = shared_macro_constraints
        all_group_ids = all_group_ids

        problem = {
            'name': 'global_problem',
            'original_objective_function': original_objective_function,
            'objective_function': objective_function,
            'batch_objective_function': batch_objective_function,
            'inequality_constraints': inequality_constraints,
            'macro_constraints_list': macro_constraints,
            'shared_macro_constraints': shared_macro_constraints,
            'all_group_ids': all_group_ids,
            'aggregation_teachers_list': [],
            'num_classes': self.num_classes,
            'performance_constraint': self.performance_constraint,
            'performance_step': self.performance_step,
        }
        debug_print('All group ids: ', all_group_ids)
        debug_print('Macro constraints: ', macro_constraints)
        debug_print('Num of macro constraints: ', len(macro_constraints))
        # print('Inequality constraints: ', inequality_constraints)
        debug_print('Num of inequality constraints: ', len(inequality_constraints))
        return problem

    def _select_clients(self, fraction=1.0):
        """
        Sample clients for the current federated round.

        Args:
            fraction: Fraction of available clients to sample.

        Returns:
            Tuple ``(selected_indices, selected_clients)``.
        """
        assert 0 < fraction <= 1, "Fraction must be between 0 and 1"
        num_clients = max(1, math.ceil(len(self.clients) * fraction))
        selected_indices = random.sample(range(len(self.clients)), num_clients)
        selected_clients = [self.clients[i] for i in selected_indices]
        debug_print(
            f'[Server] Selected clients {[s+1 for s in selected_indices]} for this round')
        return selected_indices, selected_clients

    def _broadcast_fn(self, fn_name, **kwargs):
        """
        Invoke a named RPC method on a set of clients.

        Args:
            fn_name: Client method name to call.
            **kwargs: Arguments forwarded to each client. Optional
                ``selected_clients`` overrides broadcasting to all clients.

        Returns:
            List of resolved Ray results in client order.
        """
        assert isinstance(fn_name, str), "fn_name must be a string"
        selected_clients = kwargs.get('selected_clients', self.clients)
        handlers = []
        results = []

        for client in selected_clients:
            assert hasattr(
                client, fn_name), f"Client does not have {fn_name} method"
            handlers.append(getattr(client, fn_name).remote(**kwargs))
        for handler in handlers:
            results.append(ray.get(handler))
        return results

    def setup(self, **kwargs):
        """
        Instantiate clients and initialize per-client local model state.

        Args:
            **kwargs: Optional setup payload kept for interface compatibility.
        """
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, self.checkpoint_name)
        self.clients = self._create_clients(
            self.clients_init_fn_list)
        self.client_model_params = [
            copy.deepcopy(self.model.state_dict())
            for _ in self.clients
        ]
        self._broadcast_fn('setup',
                           global_model_ckpt_path=self.checkpoint_path)

    def save(self, metrics, path):
        """
        Save the current global model and associated metrics.

        Args:
            metrics: Metric dictionary to store with the checkpoint.
            path: Destination checkpoint path.
        """
        result_to_save = {
            'model_params': self.model.state_dict(),
            'metrics': metrics
        }
        torch.save(result_to_save, path)

    def _build_client_problem(self, client_idx, local_teacher_models):
        """Build one client's local problem for the current server round.

        Every client starts from scratch. Its performance reference ``p*`` is
        initialized by the first validation evaluation and can only increase.
        """
        current_problem = copy.deepcopy(self.problem)
        current_problem['performance_step'] = self.performance_step
        current_problem['aggregation_teachers_list'] = copy.deepcopy(
            local_teacher_models)
        return current_problem

    def step(self, **kwargs):
        """
        Execute one global FedFairLab round.

        The round samples a teacher from history, launches client local updates,
        distills the ensemble target into a new global model, updates global
        history, logs metrics, and applies server callbacks.

        Args:
            **kwargs: Optional round metadata.
        """
        handlers = []
        results = []
        selected_indices, selected_clients = self._select_clients(
            fraction=self.fraction)
        is_first_round = self.first_round
        teacher_model_params, _ = self.sample_teacher_from_history()
        teacher_sampling_diagnostics = copy.deepcopy(
            self._last_teacher_sampling_diagnostics)
        local_teacher_models = [] if is_first_round else [
            copy.deepcopy(teacher_model_params)]

        for client_idx, client in zip(selected_indices, selected_clients):
            current_problem = self._build_client_problem(
                client_idx, local_teacher_models)
            handlers.append(client.fit.remote(
                model_params=copy.deepcopy(
                    self.client_model_params[client_idx]),
                problem=current_problem,
                selected_clients=selected_clients,
            ))
        self.first_round = False
        for handler in handlers:
            results.append(ray.get(handler))
        for client_idx, result in zip(selected_indices, results):
            self.client_model_params[client_idx] = copy.deepcopy(
                result['params'])

        aggregated_model_params, global_eval = self.aggregation_phase(
            params=results,
            selected_clients=selected_clients,
            teacher_model_params=teacher_model_params,
        )
        global_eval['metrics'].update(teacher_sampling_diagnostics)

        self.model.load_state_dict(aggregated_model_params)
        self.update_global_history(aggregated_model_params, global_eval)

        try:
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    stop, counter = callback(metrics=global_eval['metrics'])
                    global_eval['metrics']['global_early_stopping'] = counter
                    if stop:
                        self.logger.log(global_eval['metrics'])
                        raise EarlyStoppingException
                elif isinstance(callback, ModelCheckpoint):
                    model_checkpoint = callback(save_fn=partial(self.save,
                                                                global_eval['metrics']),
                                                metrics=global_eval['metrics']
                                                )

                    global_eval['metrics']['global_checkpoint'] = 1 if model_checkpoint else 0
                    self.global_model = copy.deepcopy(self.model)
            self.logger.log(global_eval['metrics'])
            # self.model.load_state_dict(new_params)

        except EarlyStoppingException:
            raise EarlyStoppingException

    def log_final_results(self, split='test', **kwargs):
        """
        Log final global checkpoint metrics and upload the global model artifact.

        Args:
            **kwargs: Reserved for future logging options.
        """
        best_results = None
        artifact_path = None
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_results = callback.get_best_model()
                artifact_name = 'global_model'
                artifact_path = callback.get_model_path()
                self.logger.log_artifact(artifact_name,
                                         artifact_path)

        if best_results is None:
            raise ValueError("Best global checkpoint not found")

        if split == 'val':
            metric_prefix = kwargs.get('metric_prefix', 'final')
            final_scores = {
                f'{metric_prefix}_{key}': value
                for key, value in best_results['metrics'].items()
            }
        elif split == 'test':
            model_params = best_results.get('model_params', best_results)
            test_results = self.evaluate(
                model_params=model_params,
                split='test',
                log_results=False,
            )
            final_scores = {
                f'final_{key}': value
                for key, value in test_results['metrics'].items()
            }
        else:
            raise ValueError(f"Unsupported final split: {split}")
        self.logger.log(final_scores)

    def execute(self, **kwargs):
        """
        Run full server training.

        Initializes the global history with the starting model and executes
        federated rounds until the iteration budget or early stopping.

        Args:
            **kwargs: Optional execution controls.
        """

        self.first_round = True
        global_eval = self.evaluate(
            model_params=self.model.state_dict(),
            update_performance_reference=True,
        )

        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                stop, counter = callback(metrics=global_eval['metrics'])
                global_eval['metrics']['global_early_stopping'] = counter
                if stop:
                    self.logger.log(global_eval)
                    raise EarlyStoppingException
            elif isinstance(callback, ModelCheckpoint):
                model_checkpoint = callback(save_fn=partial(self.save,
                                                            global_eval['metrics']),
                                            metrics=global_eval['metrics']
                                            )
                global_eval['metrics']['global_checkpoint'] = 1 if model_checkpoint else 0

        self.logger.log(global_eval['metrics'])

        try:
            pbar = tqdm(
                range(self.num_federated_iterations),
                desc="Global rounds",
                unit="round",
                disable=not self.verbose,
            )
            for i in pbar:
                pbar.set_postfix_str(f"Global Round {i+1}")
                self.step(round=i)
        except EarlyStoppingException:
            pass
        debug_print('End of the global rounds')

    def evaluate_model_from_ckpt(self, **kwargs):
        """
        Evaluate a model checkpoint globally or on one client.

        Args:
            **kwargs: Requires ``checkpoint_path``. Optional ``client_id``
                restricts evaluation to a single client. ``split`` defaults to
                ``test``.

        Returns:
            Dictionary of final-prefixed metric values.
        """
        checkpoint_path = kwargs.get('checkpoint_path')
        client_id = kwargs.get('client_id')
        split = kwargs.get('split', 'test')
        assert checkpoint_path is not None, "Checkpoint Path is required"

        model_params = load_trusted_checkpoint(checkpoint_path)

        if client_id is None:
            try:
                self.model.load_state_dict(model_params['model_params'])
            except KeyError:
                self.model.load_state_dict(model_params)
        else:
            try:
                model_params = model_params['model_state_dict']
                if isinstance(model_params, dict):
                    self.model.load_state_dict(model_params)
                else:
                    self.model = copy.deepcopy(model_params)
            except KeyError:
                self.model.load_state_dict(model_params)

        debug_print('Evaluating model from checkpoint:', checkpoint_path)

        global_scores = self.evaluate(model_params=self.model.state_dict(),
                                      client_id=client_id,
                                      split=split,
                                      log_results=False)
        final_scores = {}
        for key, v in global_scores['metrics'].items():
            final_scores[f'final_{key}'] = v
        # print('Final scores:',final_scores)
        return final_scores

    def evaluate_saved_checkpoints_on_test(self):
        """Evaluate the saved best global checkpoint without running training."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Global checkpoint not found: {self.checkpoint_path}")

        final_scores = self.evaluate_model_from_ckpt(
            checkpoint_path=self.checkpoint_path,
            split='test',
        )
        self.logger.log(final_scores)
        self.logger.log_artifact('global_model', self.checkpoint_path)
        return final_scores

    def fine_tune(self, **kwargs):
        """
        Broadcast optional fine-tuning using the saved global checkpoint.

        Args:
            **kwargs: Reserved fine-tuning options.
        """
        handlers = []
        if os.path.exists(self.checkpoint_path):
            global_model = load_trusted_checkpoint(self.checkpoint_path)
            self.model.load_state_dict(
                global_model.get('model_params', global_model)
            )

        self._broadcast_fn('fine_tune', global_model=self.model)
        return

    def shutdown(self, **kwargs):
        """
        Close server and client resources.

        Args:
            **kwargs: ``log_results`` controls final global logging.
        """
        log_results = kwargs.get('log_results', True)
        log_global_results = kwargs.get('log_global_results', log_results)
        log_client_results = kwargs.get('log_client_results', log_results)
        final_split = kwargs.get('final_split', 'test')
        if log_global_results:
            self.log_final_results(
                split=final_split,
                metric_prefix=kwargs.get('metric_prefix', 'final'),
            )

        self.logger.close()
        self._broadcast_fn('shutdown',
                           log_results=log_client_results,
                           final_split=final_split,
                           metric_prefix=kwargs.get('metric_prefix', 'final'))
