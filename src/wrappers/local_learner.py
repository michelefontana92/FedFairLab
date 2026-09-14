from debug_utils import debug_print
from .torch_nn_wrapper import TorchNNWrapper
import torch
import tqdm
from callbacks import EarlyStopping, ModelCheckpoint
import os
from entmax import entmax_bisect
from metrics import Performance, GroupFairnessMetric
import copy
import time


class EarlyStoppingException(Exception):
    """Raised internally when a local learner meets an early-stopping criterion."""
    pass


class LocalLearner(TorchNNWrapper):
    """
    LocalLearner is a class that implements a local learner with ALM optimization.
    Methods:
        compute_groups_cardinality(): Computes the cardinality of groups.
        _init_inequality_lambdas(): Initializes the inequality lambdas.
        _init_alm_parameters(): Initializes the ALM parameters.
        update_lambdas_inequality(constraints): Updates the inequality lambdas.
        update_lambdas_equality(constraints): Updates the equality lambdas.
        update_alm_parameters_and_metrics(update_alm=True, **kwargs): Updates the ALM parameters and computes metrics.
        compute_constraints(**kwargs): Computes the constraints.
        compute_score(**kwargs): Computes the score.
        compute_loss_fn(**kwargs): Computes the loss function.
        _compute_metrics(metrics, prefix='val', **kwargs): Computes metrics.
        _training_step(batch, batch_idx): Performs a training step.
        _train_eval_step(**kwargs): Performs a training evaluation step.
        _validation_step(**kwargs): Performs a validation step.
        set_constraints(inequality_constraints_fn_list, equality_constraints_fn_list, macro_constraints_list, inequality_lambdas, equality_lambdas): Sets the constraints.
        evaluate(model_dict, **kwargs): Evaluates the model.
        compute_val_kwargs(model_dict, use_training=False): Computes validation kwargs.
        compute_violations(val_kwargs, **kwargs): Computes the violations.
        fit(**kwargs): Fits the model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a local learner with ALM state, objectives, metrics and caches.

        Args:
            *args: Forwarded to ``TorchNNWrapper``.
            **kwargs: Model, optimizer factory, constraints, surrogate
                objectives, callbacks, data module, and teacher configuration.
        """

        super(LocalLearner, self).__init__(*args, **kwargs)
        self.id = kwargs.get('id', 'LagrangianWrapper')
        self.compute_only_score = kwargs.get('compute_only_score', False)
        self.optimizer_fn: callable = kwargs.get('optimizer_fn')
        self.lagrangian_checkpoints = kwargs.get('lagrangian_checkpoints', [])
        # self.training_group_name: str = kwargs.get('training_group_name')

        self.teacher_model = kwargs.get('teacher_model')
        # self.distillation_loss_fn:callable = kwargs.get('distillation_loss_fn')
        self.batch_objective_function = kwargs.get('batch_objective_fn')
        self.original_objective_fn: callable = kwargs.get(
            'original_objective_fn')
        self.objective_fn: callable = kwargs.get('objective_fn')
        self.inequality_constraints_fn_list: list = kwargs.get(
            'inequality_constraints')
        self.equality_constraints_fn_list: list = kwargs.get(
            'equality_constraints')

        self.mu_max = kwargs.get('mu_max', 1e3)
        self.nu_max = kwargs.get('nu_max', 100)
        self.lambda_equality_max = kwargs.get('lambda_equality_max', 100)
        self.lambda_inequality_max = kwargs.get('lambda_inequality_max', 100)

        self.rho = kwargs.get('rho', 2)
        self.mu_0 = kwargs.get('mu_0', 2)
        # Valore di damping per rallentare l'aggiornamento
        self.damping_factor = kwargs.get('damping_factor', 1.0)

        self.gamma_objective = kwargs.get('gamma_objective', 0.8)
        self.performance_budget = kwargs.get('performance_budget')

        self.inequality_lambdas_0_value = kwargs.get(
            'inequality_lambdas_0_value', 0.1)
        self.equality_lambdas_0_value = kwargs.get(
            'equality_lambdas_0_value', 0.)
        self.objective_multiplier_0_value = kwargs.get(
            'objective_multiplier_0_value', 1)
        self.macro_constraints_list = kwargs.get('macro_constraints_list')
        # Assicurati che tutti i tensori siano su device
        self.inequality_lambdas_0 = torch.ones(len(
            self.inequality_constraints_fn_list), device=self.device) * self.inequality_lambdas_0_value
        self.equality_lambdas_0 = torch.ones(len(
            self.equality_constraints_fn_list), device=self.device) * self.equality_lambdas_0_value
        self.objective_multiplier_0 = torch.tensor(
            self.objective_multiplier_0_value, device=self.device)
        self.lambda0_max_value = kwargs.get('lambda0_max_value', 0.1)
        self.target_groups = set()
        self.all_group_ids = kwargs.get('all_group_ids')

        self.teachers_kwargs = {'train': {},
                                'val': {}
                                }

        self.verbose = kwargs.get('verbose', False)
        # self.compute_groups_cardinality()
        self.state_path = None
        self.active_groups = {}
        self.teacher_model_list = kwargs.get('teacher_model_list', [])
        for constraint in self.inequality_constraints_fn_list:
            if constraint.group_name is not None:
                self.target_groups.add(constraint.group_name)
                if constraint.group_name not in self.active_groups:
                    self.active_groups[constraint.group_name] = []
                for c in constraint.target_groups:
                    if c not in self.active_groups[constraint.group_name]:
                        self.active_groups[constraint.group_name].append(
                            c.item())

        assert self.macro_constraints_list is not None, f'{self.macro_constraints_list} has to be provided'
        self.group_cardinality = None
        # print('Device:', self.device)
        # print('Device:',self.device)
        self._init_alm_parameters()
        self.all_groups_name = [
            group_name for group_name in self.target_groups]
        # print(f'[INFO] Inequality constraints: {self.inequality_constraints_fn_list}')
        # print(f'[INFO] Macro constraints: {self.macro_constraints_list}')

        def init_weights(m):
            """Initialize linear layers with Xavier weights and zero biases."""
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.model.apply(init_weights)
        self.subtract_upper_bound = kwargs.get('subtract_upper_bound', True)

    def reset(self):
        """
        Reset learner callbacks before a new training run.
        """
        for checkpoint in self.checkpoints:
            checkpoint.reset()
        for checkpoint in self.lagrangian_checkpoints:
            checkpoint.reset()

    def set_teachers_kwargs(self, teachers_kwargs):
        """
        Set cached teacher tensors used by objective/distillation losses.

        Args:
            teachers_kwargs: Dictionary containing teacher logits,
                probabilities, softmax outputs and predictions.
        """
        # print('Setting Teachers kwargs:',teachers_kwargs)
        self.teachers_kwargs.update(teachers_kwargs)

    def set_ensemble_teacher_logits(self, ensemble_logits):
        """
        Register a precomputed ensemble target as the sole distillation teacher.

        Args:
            ensemble_logits: Tensor shaped ``[num_examples, num_classes]``,
                ``[1, num_examples, num_classes]`` or a dictionary with
                ``train`` and ``val`` tensors. It is converted to the teacher
                cache format expected by the surrogate losses.
        """
        if ensemble_logits is None:
            self.teachers_kwargs['train'] = {}
            self.teachers_kwargs['val'] = {}
            return

        if isinstance(ensemble_logits, dict):
            split_logits = ensemble_logits
        else:
            split_logits = {'train': ensemble_logits}

        for split, logits in split_logits.items():
            if logits is None:
                self.teachers_kwargs[split] = {}
                continue
            logits = logits.detach().to(self.device)
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            self.teachers_kwargs[split] = {
                'teacher_logits_list': logits,
                'teacher_softmax_list': torch.softmax(logits, dim=-1),
                'teacher_probabilities': torch.softmax(logits, dim=-1),
                'teacher_predictions_list': torch.argmax(logits, dim=-1),
            }

    def compute_weighted_ensemble_logits(self, teacher_model_dict_list, weights, use_training=True):
        """
        Compute weighted logits for a list of teacher models on local data.

        Args:
            teacher_model_dict_list: Candidate model state dicts.
            weights: Ensemble weights. They are normalized before use.
            use_training: If true, compute logits on the training split;
                otherwise use validation data.

        Returns:
            Tensor shaped ``[num_examples, num_classes]`` containing the weighted
            ensemble logits for this client.
        """
        if teacher_model_dict_list is None or len(teacher_model_dict_list) == 0:
            raise ValueError(
                "teacher_model_dict_list must contain at least one model")

        weights_tensor = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device)
        if weights_tensor.numel() != len(teacher_model_dict_list):
            raise ValueError(
                "weights must match teacher_model_dict_list length")
        weights_tensor = weights_tensor / weights_tensor.sum().clamp(min=1e-12)

        loader = self.data_module.train_loader_eval(
            batch_size=None) if use_training else self.data_module.val_loader(batch_size=512)
        weighted_logits_list = []

        models = []
        for teacher_model_dict in teacher_model_dict_list:
            teacher_model = copy.deepcopy(self.model)
            teacher_model.load_state_dict(copy.deepcopy(teacher_model_dict))
            teacher_model.to(self.device)
            teacher_model.eval()
            models.append(teacher_model)

        for batch in loader:
            inputs = batch['data'].float().to(self.device)
            batch_logits = None
            with torch.no_grad():
                for weight, teacher_model in zip(weights_tensor, models):
                    outputs = teacher_model(inputs)
                    weighted_outputs = weight * outputs
                    batch_logits = weighted_outputs if batch_logits is None else batch_logits + weighted_outputs
            weighted_logits_list.append(batch_logits.detach().cpu())

        return torch.cat(weighted_logits_list, dim=0)

    def evaluate_precomputed_logits(self, logits, split='val'):
        """Evaluate externally supplied logits with the learner's metric objects."""
        if split != 'val':
            raise ValueError(
                'Precomputed ensemble monitoring currently supports validation only')
        loader = self.data_module.val_loader(batch_size=512)
        labels_list = []
        groups_lists = {}
        for batch in loader:
            labels_list.append(batch['labels'].detach().cpu())
            for group_name, group_values in batch['groups'].items():
                groups_lists.setdefault(group_name, []).append(
                    group_values.detach().cpu())

        labels = torch.cat(labels_list, dim=0)
        group_ids = {
            group_name: torch.cat(values, dim=0)
            for group_name, values in groups_lists.items()
        }
        logits = logits.detach().cpu()
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                'Ensemble logits and validation labels must have equal length')
        return self._compute_metrics(
            self.metrics,
            prefix=split,
            predictions=torch.argmax(logits, dim=1),
            labels=labels,
            group_ids=group_ids,
        )

    def compute_groups_cardinality(self):
        """
        Compute observed group cardinalities on the training set.
        """
        groups = next(iter(self.get_train_loader_eval()))['groups']
        self.group_cardinality = {group_name: {}
                                  for group_name in self.target_groups}
        self.max_cardinality = {
            group_name: 0 for group_name in self.target_groups}
        for group_name in self.target_groups:
            for group in groups[group_name].unique():
                self.group_cardinality[group_name].update(
                    {group.item(): len(groups[group_name][groups[group_name] == group])})
                if len(groups[group_name][groups[group_name] == group]) > self.max_cardinality[group_name]:
                    self.max_cardinality[group_name] = len(
                        groups[group_name][groups[group_name] == group])

    def _init_inequality_lambdas(self):
        """
        Initialize inequality Lagrange multipliers.
        """
        self.inequality_lambdas = torch.ones_like(
            self.inequality_lambdas_0, device=self.device) * self.inequality_lambdas_0_value

    def _init_alm_parameters(self):
        """
        Initialize all augmented-Lagrangian parameters.
        """
        self._init_inequality_lambdas()
        self.equality_lambdas = self.equality_lambdas_0
        self.mu = self.mu_0
        self.objective_multiplier = self.objective_multiplier_0

    def update_lambdas_inequality(self, constraints):
        """
        Update inequality Lagrange multipliers.

        Args:
            constraints: Tensor of inequality constraint violations.

        Returns:
            Updated multiplier tensor.
        """

        if constraints is None:
            return self.inequality_lambdas

        new_lambdas = torch.max(
            torch.ones_like(self.inequality_lambdas,
                            device=self.device)*self.inequality_lambdas_0_value,
            self.inequality_lambdas + self.mu * torch.max(constraints, torch.zeros_like(constraints,
                                                                                        device=self.device))
        )

        assert torch.all(
            new_lambdas >= self.inequality_lambdas_0_value), 'Negative Lagrange multipliers!'
        return new_lambdas

    def update_lambdas_equality(self, constraints):
        """
        Update equality Lagrange multipliers.

        Args:
            constraints: Tensor of equality constraint values.

        Returns:
            Updated multiplier tensor.
        """

        if constraints is None:
            return self.equality_lambdas
        new_lambdas = self.equality_lambdas + self.mu * constraints * self.damping_factor
        new_lambdas = torch.clamp(
            new_lambdas, min=self.equality_lambdas_0_value, max=self.lambda_equality_max)

        return new_lambdas

    def update_alm_parameters_and_metrics(self, update_alm=True, split='val', **kwargs):
        """
        Evaluate metrics and optionally update ALM multipliers.

        Args:
            update_alm: If true, compute training violations and update
                Lagrange multipliers before scoring validation data.
            split: Evaluation split. Must be ``val`` or ``test``.
            **kwargs: Optional cached train/evaluation payloads.

        Returns:
            Metric dictionary prefixed with the selected evaluation split.
        """
        metrics = {}
        self.model.eval()
        with torch.no_grad():
            if split not in {'val', 'test'}:
                raise ValueError(f"Unsupported evaluation split: {split}")
            if update_alm and split != 'val':
                raise ValueError(
                    "Test data cannot be used while updating training/ALM state")

            eval_kwargs = kwargs.get(f'{split}_kwargs')
            # Preserve compatibility with callers that precompute validation data.
            if eval_kwargs is None and split == 'val':
                eval_kwargs = kwargs.get('val_kwargs')
            if eval_kwargs is None:
                eval_loader = (
                    self.data_module.val_loader(batch_size=512)
                    if split == 'val'
                    else self.data_module.test_loader(batch_size=512)
                )
                eval_kwargs = self._compute_kwargs_in_batches(
                    eval_loader,
                    self.model,
                    use_entmax=False,
                    use_training=False,
                    evaluation_split=split,
                )
                # print(f'[INFO] Probabilities: {val_kwargs["probabilities"][:5,:]}')
                # print(f'[INFO] Inequality constraints: {val_kwargs["inequality_constraints"]}')
            if update_alm:
                train_loader = self.data_module.train_loader_eval(
                    batch_size=None)
                train_kwargs = self._compute_kwargs_in_batches(
                    train_loader, self.model, use_entmax=False, use_training=True)
                kwargs['train_kwargs'] = train_kwargs
                inequality_constraints = train_kwargs['inequality_constraints']
                equality_constraints = train_kwargs['equality_constraints']
                self._apply_early_stopping(
                    inequality_constraints, equality_constraints)

                if inequality_constraints is not None:
                    inequality_constraints = inequality_constraints * self.inequality_mask

                if equality_constraints is not None:
                    equality_constraints = equality_constraints * self.equality_mask

                if inequality_constraints is not None:
                    self.inequality_lambdas = self.update_lambdas_inequality(
                        inequality_constraints)
                if equality_constraints is not None:
                    self.equality_lambdas = self.update_lambdas_equality(
                        equality_constraints)

            eval_score = self.compute_score(**eval_kwargs)
            metrics[f'{split}_constraints_score'] = eval_score
            eval_loss = self.compute_loss_fn(**eval_kwargs)
            metrics[f'{split}_loss'] = eval_loss
            if not self.compute_only_score:
                metrics.update(self._compute_metrics(
                    self.metrics, prefix=split, **eval_kwargs))
            return metrics

    def _apply_early_stopping(self, inequality_constraints, equality_constraints):
        """
        Apply per-constraint early stopping masks for ALM updates.

        Args:
            inequality_constraints: Current inequality violations.
            equality_constraints: Current equality violations.
        """
        n_inequality_constraints = len(self.inequality_constraints_fn_list)
        self.inequality_mask = torch.ones_like(
            self.inequality_lambdas, device=self.device)
        self.equality_mask = torch.ones_like(
            self.equality_lambdas, device=self.device)
        cached_scores = {}
        for i, checkpoint in enumerate(self.lagrangian_checkpoints):
            if isinstance(checkpoint, EarlyStopping):
                if i < n_inequality_constraints:
                    if i not in cached_scores:
                        cached_scores[i] = {
                            'violations': inequality_constraints[i]}
                    update, _ = checkpoint(metrics=cached_scores[i])
                    if not update:
                        # Ferma l'aggiornamento per questo vincolo
                        self.inequality_mask[i] = 0
                    else:
                        checkpoint.reset(keep_best=True)
                else:
                    eq_index = i - n_inequality_constraints
                    if eq_index not in cached_scores:
                        cached_scores[eq_index] = {
                            'violations': equality_constraints[eq_index]}
                    update, _ = checkpoint(metrics=cached_scores[eq_index])
                    if not update:
                        # Ferma l'aggiornamento per questo vincolo
                        self.equality_mask[eq_index] = 0
                    else:
                        checkpoint.reset()

    def compute_constraints(self, **kwargs):
        """
        Evaluate all configured equality and inequality constraints.

        Args:
            **kwargs: Tensor payload containing logits, labels, groups and masks.

        Returns:
            Tuple ``(inequality_constraints, equality_constraints)``.
        """
        device = self.device
        if len(self.inequality_constraints_fn_list) > 0:
            inequality_constraints = torch.stack(
                [torch.clamp(constraint_fn(**kwargs), min=0) for constraint_fn in self.inequality_constraints_fn_list], dim=0
            ).to(device)
        else:
            inequality_constraints = torch.tensor([], device=device)

        if len(self.equality_constraints_fn_list) > 0:
            equality_constraints = torch.stack(
                [constraint_fn(**kwargs) for constraint_fn in self.equality_constraints_fn_list], dim=0
            ).to(device)
        else:
            equality_constraints = torch.tensor([], device=device)

        return inequality_constraints, equality_constraints

    def compute_score(self, **kwargs):
        """
        Compute the local selection score from objective and constraint penalties.

        Args:
            **kwargs: Payload containing objective value and constraint tensors.

        Returns:
            Scalar score tensor.
        """
        objective_function = kwargs.get('original_objective_function')
        inequality_constraints = kwargs.get('inequality_constraints')
        equality_constraints = kwargs.get('equality_constraints')
        # print('Original_objective function',objective_function.item())
        # Inizializza lo score con la funzione obiettivo

        score = objective_function.clone()
        # print(f'[INFO] Score before constraints: {score.item()}')
        # Inizializza una variabile per il conteggio delle violazioni dei vincoli
        total_penalty = 0
        # Performance improvement and budget remain separate soft constraints
        # in the Lagrangian.  Model selection scores only the budget violation:
        # the improvement constraint guides optimization without rewarding the
        # selector for performance beyond the admissible lower bound.
        if (
            self.performance_budget is not None
            and len(inequality_constraints) > 1
            and len(self.macro_constraints_list) > 0
            and 0 in self.macro_constraints_list[0]
            and 1 in self.macro_constraints_list[0]
        ):
            total_penalty += torch.clamp(
                inequality_constraints[[1]], min=0
            ).sum()

        if len(inequality_constraints) > 0:
            for macro_idx, macro_constraint in enumerate(self.macro_constraints_list):
                if (
                    self.performance_budget is not None
                    and macro_idx == 0
                    and 0 in macro_constraint
                    and 1 in macro_constraint
                ):
                    continue
                if len(macro_constraint) > 0:
                    inequality_penalty = torch.max(torch.clamp(
                        inequality_constraints[macro_constraint], min=0))
                    total_penalty += inequality_penalty
                    # print(f'[INFO] Inequality Penalty for macro constraint {macro_constraint}: {inequality_penalty.item()}')

        if len(equality_constraints) > 0:
            equality_penalty = torch.max(torch.abs(equality_constraints))
            total_penalty += equality_penalty
        score -= total_penalty
        # print(f'Final Score after constraints: {score.item()}')
        # if total_penalty > 0:
        #    print('Total Penalty:',total_penalty)
        # print('Total Penalty:',total_penalty)
        # assert score >= 0, 'Score non può essere negativo!'
        return score

    def compute_loss_fn(self, **kwargs):
        """
        Compute the ALM training loss.

        Args:
            **kwargs: Payload containing objective, batch objective, constraints
                and group identifiers.

        Returns:
            Scalar loss tensor.
        """
        loss = kwargs['objective_function'].clone()
        if torch.isnan(loss).any():
            raise ValueError("NaN nella loss!")

        batch_objective_function = kwargs['batch_objective_function']
        group_ids = kwargs.get('group_ids', {})
        equality_constraints = kwargs.get('equality_constraints')
        inequality_constraints = kwargs.get('inequality_constraints')

        if self.target_groups and group_ids:
            group_losses = []
            total_weight = 0.0
            for group_name in self.target_groups:
                group_tensor = group_ids[group_name]
                for group_id in torch.unique(group_tensor):
                    mask = group_tensor == group_id
                    if mask.sum() > 0:
                        group_loss = batch_objective_function[mask].mean()
                        weight = 1.0 - mask.sum().item() / \
                            batch_objective_function.shape[0]
                        group_losses.append(weight * group_loss)
                        total_weight += weight
            if group_losses and total_weight > 0:
                loss += torch.stack(group_losses).sum() / total_weight

        if equality_constraints is not None and len(self.equality_constraints_fn_list) > 0:
            penalty = torch.mean(torch.abs(equality_constraints)) * self.mu
            lagrange = (self.equality_lambdas * equality_constraints).sum()
            loss += penalty + lagrange

        if inequality_constraints is not None and len(self.inequality_constraints_fn_list) > 0:
            if torch.any(self.inequality_lambdas > 0):
                penalty = torch.sum(torch.clamp(inequality_constraints, min=0))
                lambdas_updated = torch.clamp(
                    self.inequality_lambdas + self.mu * inequality_constraints, min=0)
                lagrange = (lambdas_updated.pow(
                    2) - self.inequality_lambdas.pow(2)).sum() / (2 * self.mu)
                loss += penalty + lagrange

        if torch.isnan(loss).any():
            raise ValueError("NaN nella loss finale!")

        return loss

    def _compute_kwargs_in_batches(self, loader, model, use_entmax=True,
                                   use_training=False, evaluation_split=None):
        """
        Build objective/metric payloads by evaluating a full loader.

        Args:
            loader: DataLoader to iterate.
            model: Model used to compute logits.
            use_entmax: Whether to use Entmax probabilities.
            use_training: Selects train or validation teacher caches.

        Returns:
            Dictionary containing concatenated logits, labels, groups,
            predictions, probabilities, objectives and constraints.
        """
        all_logits = []
        all_labels = []
        all_indices = []
        all_positive_masks = []
        if self.all_group_ids is not None:
            all_group_ids = {group_name: []
                             for group_name in self.all_group_ids.keys()}
            all_group_ids_list = {group_name: []
                                  for group_name in self.all_group_ids.keys()}
            # print("[INFO] Collected all group IDs and their corresponding lists.")

        else:
            if self.target_groups:
                all_group_ids = {group_name: [] for group_name in loader.dataset[0]['groups'].keys(
                ) if group_name in self.target_groups}
                all_group_ids_list = {group_name: [] for group_name in loader.dataset[0]
                                      ['groups_ids_list'].keys() if group_name in self.target_groups}
            else:
                debug_print("[INFO] No target groups specified, skipping group collection.")
                all_group_ids = {}
                all_group_ids_list = {}

        for batch in loader:
            inputs = batch['data'].float().to(self.device)
            outputs = model(inputs)

            all_logits.append(outputs)
            all_labels.append(batch['labels'].to(self.device))
            all_indices.append(batch['index'])
            all_positive_masks.append(batch['positive_mask'].to(self.device))

            if self.target_groups or self.all_group_ids is not None:
                for group_name in all_group_ids:
                    all_group_ids[group_name].append(
                        batch['groups'][group_name].to(self.device))
                for group_name in all_group_ids_list:
                    all_group_ids_list[group_name].append(
                        batch['groups_ids_list'][group_name].to(self.device))

        final_logits = torch.cat(all_logits, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_indices = torch.cat(all_indices, dim=0)
        final_positive_masks = torch.cat(all_positive_masks, dim=0)

        if self.all_group_ids is not None:
            final_group_ids = {g: torch.cat(
                all_group_ids[g], dim=0) for g in all_group_ids}
            final_group_ids_list = {g: torch.cat(
                all_group_ids_list[g], dim=0) for g in all_group_ids_list}
        else:
            final_group_ids = {g: torch.cat(
                all_group_ids[g], dim=0) for g in all_group_ids} if self.target_groups else {}
            final_group_ids_list = {g: torch.cat(
                all_group_ids_list[g], dim=0) for g in all_group_ids_list} if self.target_groups else {}

        batch_dict = {
            'logits': final_logits,
            'labels': final_labels,
            'groups': final_group_ids,
            'groups_ids_list': final_group_ids_list,
            'positive_mask': final_positive_masks,
            'index': final_indices,
        }

        return self._compute_kwargs(
            batch_dict,
            final_logits,
            use_entmax=use_entmax,
            use_training=use_training,
            evaluation_split=evaluation_split,
        )

    def _compute_kwargs(self, batch, outputs, use_entmax=True,
                        use_training=False, evaluation_split=None):
        """
        Build objective/metric payloads for one mini-batch.

        Args:
            batch: Batch dictionary returned by the data module.
            outputs: Model logits for the batch.
            use_entmax: Whether to compute Entmax probabilities.
            use_training: Selects the training teacher cache.
            evaluation_split: Non-training cache name. Test evaluation uses an
                empty cache and therefore cannot consume validation teachers.

        Returns:
            Dictionary consumed by objective, metric and constraint functions.
        """
        device = self.device

        # === Gruppi (fairness): opzionali ===
        if self.all_group_ids is not None:
            group_ids = {g: batch['groups'][g].to(
                device) for g in self.all_group_ids.keys()}
            group_ids_list = {g: batch['groups_ids_list'][g].to(
                device) for g in self.all_group_ids.keys()}
        else:
            group_ids = {g: batch['groups'][g].to(device) for g in batch.get(
                'groups', {})} if self.target_groups else {}
            group_ids_list = {g: batch['groups_ids_list'][g].to(device) for g in batch.get(
                'groups_ids_list', {})} if self.target_groups else {}

        # === Mask e label: obbligatori ===
        positive_mask = batch.get('positive_mask')
        if positive_mask is None:
            raise ValueError("'positive_mask' non è presente nel batch.")
        positive_mask = positive_mask.to(device)

        labels = batch.get('labels')
        if labels is None:
            raise ValueError("'labels' non è presente nel batch.")
        labels = labels.to(device)

        # === Predizioni ===
        predictions = torch.argmax(outputs, dim=-1)

        probabilities = (
            entmax_bisect(outputs*10, alpha=1.5, dim=-1)
            if use_entmax else
            torch.nn.functional.one_hot(
                predictions, num_classes=outputs.size(-1)).float()
        )
        # print(f'[LL] Outputs head: {outputs[:5,:]}')
        # print(f'[LL] Probabilities head: {probabilities[:5,:]}')
        output_distribution = torch.nn.functional.softmax(outputs, dim=-1)
        # print(f"[INFO] Output distribution: {output_distribution[:5,:]}, Predictions shape: {predictions[:5]}")
        # === Base kwargs ===
        kwargs = {
            'group_ids': group_ids,
            'group_ids_list': group_ids_list,
            'group_masks': group_ids,
            'positive_mask': positive_mask,
            'logits': outputs,
            'labels': labels,
            'probabilities': probabilities,
            'predictions': predictions,
            'output_distribution': output_distribution,
        }

        # === Allineamento teacher se index è presente ===
        indices = batch.get('index', None)
        if indices is not None and indices.numel() > 0:
            max_index = indices.max().item()
            cache_split = 'train' if use_training else (evaluation_split or 'val')
            teacher_kwargs = self.teachers_kwargs.get(cache_split, {})
            for k, v in teacher_kwargs.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] > max_index:
                    kwargs[k] = v[:, indices, ...]
        else:
            debug_print("[INFO] No valid indices — skipping teacher alignment.")

        # es: (0, batch_size, num_classes)
        empty_shape = (0, outputs.shape[0], outputs.shape[1])
        for key in [
            'teacher_logits_list',
            'teacher_probabilities',
            'teacher_predictions_list',
            'teacher_softmax_list',
        ]:
            if key not in kwargs:
                kwargs[key] = torch.empty(empty_shape, device=device)

        # print(f"[INFO] Keys in kwargs: {list(kwargs.keys())}")
        # === Funzioni surrogate e constraint ===
        kwargs['objective_function'] = self.objective_fn(**kwargs)
        kwargs['original_objective_function'] = self.original_objective_fn(
            **kwargs)
        kwargs['batch_objective_function'] = self.batch_objective_function(
            **kwargs)
        kwargs['inequality_constraints'], kwargs['equality_constraints'] = self.compute_constraints(
            **kwargs)

        return kwargs

    def _compute_metrics(self, metrics, prefix='val', **kwargs):
        """
        Compute configured performance and fairness metrics.

        Args:
            metrics: Metric objects to evaluate.
            prefix: Prefix added to metric names.
            **kwargs: Payload containing predictions, labels and group ids.

        Returns:
            Dictionary of metric names to values.
        """
        group_ids = kwargs['group_ids']
        # print('[COMPUTE METRICS] Group ids:', group_ids.keys())
        y_pred = kwargs['predictions']
        y_true = kwargs['labels']

        tmp_result = {}
        final_result = {}

        for metric in metrics:
            metric.reset()
            if issubclass(metric.__class__, GroupFairnessMetric):
                group_ids_detached = {group_name: group_ids[group_name].detach(
                ).cpu() for group_name in group_ids.keys()}
                metric.calculate(y_pred.detach().cpu(),
                                 y_true.detach().cpu(),
                                 group_ids_detached)

            elif isinstance(metric, Performance):
                metric.calculate(y_pred.detach().cpu(),
                                 y_true.detach().cpu())
            else:
                raise ValueError(f"{metric} is an invalid metric")
            tmp_result.update(metric.get())

        for key, value in tmp_result.items():
            if prefix == '':
                final_result[key] = value
            else:
                final_result[f'{prefix}_{key}'] = value
        return final_result

    def _training_step(self, batch, batch_idx):
        """
        Run one optimizer step on a mini-batch.

        Args:
            batch: Training batch dictionary.
            batch_idx: Batch index, kept for compatibility.

        Returns:
            Scalar loss value.
        """
        self.model.train()
        inputs = batch['data'].float().to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        kwargs = self._compute_kwargs(
            batch, outputs, use_entmax=True, use_training=True)
        loss = self.compute_loss_fn(**kwargs)

        if torch.isnan(loss).any():
            raise ValueError('Loss contiene NaN!')

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return loss.item()

    def _train_eval_step(self, **kwargs):
        """
        Evaluate loss and predictions on cached training payload.

        Args:
            **kwargs: Requires ``train_kwargs``.

        Returns:
            Tuple ``(loss, logits, labels, predictions)``.
        """
        self.model.eval()
        with torch.no_grad():
            train_kwargs = kwargs['train_kwargs']
            outputs = train_kwargs['logits']
            targets = train_kwargs['labels']
            loss = self.compute_loss_fn(**train_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions

    def _validation_step(self, **kwargs):
        """
        Evaluate loss and predictions on cached validation payload.

        Args:
            **kwargs: Requires ``val_kwargs``.

        Returns:
            Tuple ``(loss, logits, labels, predictions)``.
        """
        self.model.eval()
        with torch.no_grad():
            val_kwargs = kwargs['val_kwargs']
            outputs = val_kwargs['logits']
            targets = val_kwargs['labels']
            loss = self.compute_loss_fn(**val_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions

    def set_constraints(self, inequality_constraints_fn_list, equality_constraints_fn_list, macro_constraints_list, inequality_lambdas, equality_lambdas):
        """
        Replace the learner constraint set and ALM multipliers.

        Args:
            inequality_constraints_fn_list: Inequality surrogate constraints.
            equality_constraints_fn_list: Equality surrogate constraints.
            macro_constraints_list: Macro-constraint grouping.
            inequality_lambdas: Current inequality multipliers.
            equality_lambdas: Current equality multipliers.
        """
        self.inequality_constraints_fn_list = inequality_constraints_fn_list
        self.equality_constraints_fn_list = equality_constraints_fn_list
        self.macro_constraints_list = macro_constraints_list
        self.inequality_lambdas = inequality_lambdas
        self.equality_lambdas = equality_lambdas

    def evaluate(self, model_dict, split='val', **kwargs):
        """
        Evaluate a model state without permanently changing the learner model.

        Args:
            model_dict: State dict to evaluate.
            split: Evaluation split. Must be ``val`` or ``test``.
            **kwargs: Optional cached evaluation payloads.

        Returns:
            Metric dictionary.
        """

        # Impostiamo il modello in modalità valutazione
        original_model_dict = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        metrics = self.update_alm_parameters_and_metrics(
            update_alm=False, split=split, **kwargs)
        self.model.load_state_dict(original_model_dict)

        return metrics

    def compute_eval_kwargs(self, model_dict, split='val'):
        """Compute the full payload for a validation or test split."""
        if split not in {'val', 'test'}:
            raise ValueError(f"Unsupported evaluation split: {split}")
        original_model_dict = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        loader = (
            self.data_module.val_loader(batch_size=512)
            if split == 'val'
            else self.data_module.test_loader(batch_size=512)
        )
        eval_kwargs = self._compute_kwargs_in_batches(
            loader,
            self.model,
            use_entmax=False,
            use_training=False,
            evaluation_split=split,
        )
        self.model.load_state_dict(original_model_dict)
        self.model.to(self.device)
        return eval_kwargs

    def compute_val_kwargs(self, model_dict, use_training=False):
        """
        Compute full-split payloads for a model state.

        Args:
            model_dict: State dict to evaluate.
            use_training: If true, use the training split; otherwise validation.

        Returns:
            Payload dictionary for metrics/objectives/constraints.
        """
        original_model_dict = self.model.state_dict()
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        if use_training:
            loader = self.data_module.train_loader_eval(batch_size=None)
        else:
            return self.compute_eval_kwargs(model_dict, split='val')
        kwargs = self._compute_kwargs_in_batches(
            loader, self.model, use_entmax=False, use_training=use_training)
        self.model.load_state_dict(original_model_dict)
        self.model.to(self.device)
        return kwargs

    def compute_violations(self, val_kwargs, **kwargs):
        """
        Summarize constraint violations for scoring and reporting.

        Args:
            val_kwargs: Payload containing logits, labels, groups and objectives.
            **kwargs: Reserved for compatibility.

        Returns:
            Dictionary with per-group, per-constraint and macro violations.
        """
        inequality_constraints, equality_constraints = self.compute_constraints(
            **val_kwargs)
        results = {}

        violations = {k: None for k, _ in enumerate(
            self.macro_constraints_list)}

        violations_per_group_list = {}
        violations_per_group = {}
        for key, value_dict in self.group_cardinality.items():
            violations_per_group_list[key] = {k: [] for k in value_dict.keys()}

        for i, constraint_violation in enumerate(inequality_constraints):
            constraint = self.inequality_constraints_fn_list[i]
            target_groups = constraint.target_groups
            group_name = constraint.group_name
            if group_name is not None:
                for group in target_groups:
                    try:
                        violations_per_group_list[group_name][group.item()].append(
                            constraint_violation)
                    except KeyError:
                        # Manteniamo il fallback originale, ma inizializziamo a lista -> niente RuntimeError
                        if group_name not in violations_per_group_list:
                            violations_per_group_list[group_name] = {}
                        if group.item() not in violations_per_group_list[group_name]:
                            violations_per_group_list[group_name][group.item()] = [
                            ]
                        violations_per_group_list[group_name][group.item()].append(
                            constraint_violation)

        # --- FIX MINIMALE: calcolo per-dizionario, nessun try/except che schiaccia a scalare ---
        for key, value_dict in violations_per_group_list.items():
            per_group = {}
            for gid, v in value_dict.items():
                if len(v) > 0:
                    per_group[gid] = torch.stack(v).max().item()
                else:
                    # nessun contributo per quel group_id: lasciamo 0.0 come default
                    per_group[gid] = 0.0
            violations_per_group[key] = per_group
        # --------------------------------------------------------------------------------------

        results['violations_per_group'] = copy.deepcopy(violations_per_group)

        for i, macro_constraint in enumerate(self.macro_constraints_list):
            violations[i] = inequality_constraints[macro_constraint].detach(
            ).cpu().numpy()

        results['inequality_constraints_violations'] = inequality_constraints.detach(
        ).cpu().numpy()

        macro_constraints_violation = copy.deepcopy(violations)
        for i, _ in enumerate(self.macro_constraints_list):
            if len(macro_constraints_violation[i]) > 0:
                macro_constraints_violation[i] = [
                    macro_constraints_violation[i].max()]
            else:
                macro_constraints_violation[i] = []
        results['macro_constraints_violations'] = copy.deepcopy(
            macro_constraints_violation)

        return results

    def _progress_bar(self, iterable, **kwargs):
        """
        Return a tqdm progress bar only when verbose mode is enabled.

        Args:
            iterable: Iterable to wrap.
            **kwargs: tqdm configuration.

        Returns:
            Either ``tqdm(iterable)`` or the original iterable.
        """
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        else:
            return iterable

    def fit(self, **kwargs):
        """
        Train the learner with ALM optimization.

        Args:
            **kwargs: Optional ``num_epochs``, ``start_model_dict``,
                ``teacher_model_list`` and logging/evaluation flags.

        Returns:
            Tuple ``(model_state, inequality_lambdas, equality_lambdas)`` for
            the best checkpointed model.
        """
        # print('Current lambdas:',self.inequality_lambdas)
        if self.verbose:
            debug_print(f'[{self.id}]:Number of inequality constraints:',
                  len(self.inequality_constraints_fn_list))
            debug_print(f'Macro constraints:', self.macro_constraints_list)
        num_epochs = kwargs.get('num_epochs', -1)
        disable_log = kwargs.get('disable_log', False)
        evaluate_best_model = kwargs.get('evaluate_best_model', False)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs

        self.teacher_model_list = kwargs.get('teacher_model_list', [])
        # print(f'[LL {self.id}]:Number of teacher models:',len(self.teacher_model_list))
        start_model_dict = kwargs.get('start_model_dict')
        # print('[LL {self.id}]:Length of teacher model list:',len(self.teacher_model_list))

        if start_model_dict is not None:
            self.model.load_state_dict(copy.deepcopy(start_model_dict))

        # print(f'[{self.id}]:Training model with parameters:',self.model.state_dict()['fc1.weight'][:5,:])
        self.model.to(self.device)

        """
        metrics = self.update_alm_parameters_and_metrics(update_alm=True) 
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, EarlyStopping):
                stop, counter = checkpoint(metrics=metrics)
                metrics['early_stopping'] = counter
                if stop:
                    if not disable_log:
                        self.logger.log(metrics)
                    raise EarlyStoppingException

            elif isinstance(checkpoint, ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        
        if not disable_log:
            self.logger.log(metrics)
        """
        self.model.train()
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.final_inequality_lambdas = copy.deepcopy(self.inequality_lambdas)
        self.final_equality_lambdas = copy.deepcopy(self.equality_lambdas)

        try:
            start = time.time()
            for epoch in self._progress_bar(range(n_rounds), desc=f'Epoch 0/{n_rounds}', total=n_rounds, unit='epoch'):
                train_loader = self.data_module.train_loader()
                batch_iterator = self._progress_bar(
                    train_loader, desc=f'Epoch {epoch+1}/{n_rounds}', leave=False)
                for batch_idx, batch in enumerate(batch_iterator):
                    self._training_step(batch, batch_idx)

                with torch.no_grad():
                    # metrics = self.update_alm_parameters_and_metrics(update_alm=False,**kwargs)
                    metrics = self.update_alm_parameters_and_metrics(
                        update_alm=True, **kwargs)

                    # Early stopping e model checkpoint
                    for checkpoint in self.checkpoints:
                        if isinstance(checkpoint, EarlyStopping):

                            stop, counter = checkpoint(metrics=metrics)
                            metrics['early_stopping'] = counter
                            if stop:
                                if not disable_log:
                                    self.logger.log(metrics)
                                raise EarlyStoppingException

                        elif isinstance(checkpoint, ModelCheckpoint):

                            model_checkpoint = checkpoint(
                                save_fn=self.save, metrics=metrics)
                            metrics['model_checkpoint'] = 1 if model_checkpoint else 0
                            if model_checkpoint:
                                self.final_inequality_lambdas = copy.deepcopy(
                                    self.inequality_lambdas)
                                self.final_equality_lambdas = copy.deepcopy(
                                    self.equality_lambdas)

                    if not disable_log:
                        self.logger.log(metrics)

                    if self.verbose and hasattr(batch_iterator, 'set_description'):
                        batch_iterator.set_description(
                            f'Epoch {epoch+1}/{n_rounds}')
        except EarlyStoppingException:
            pass

        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                if os.path.exists(checkpoint.get_model_path()):
                    self.load(checkpoint.get_model_path())

        end = time.time()
        debug_print(f'[{self.id}]:Training completed in {end - start:.2f} seconds.')
        """
        if evaluate_best_model:
            self.model.eval()
           
            metrics = self.update_alm_parameters_and_metrics(update_alm=False,**kwargs)
            final_metrics = {f'final_{name}': value for name, value in metrics.items()}
            if not disable_log:
               self.logger.log(final_metrics)
        """
        # print('Final lambdas:',self.inequality_lambdas)
        return copy.deepcopy(self.model.state_dict()), self.final_inequality_lambdas, self.final_equality_lambdas

    def query_teachers(self, teacher_model_dict_list, use_training=False, **kwargs):
        """
        Evaluate teacher models on local data and cache their outputs.

        Args:
            teacher_model_dict_list: Teacher model state dicts.
            use_training: If true, use the training split; otherwise validation.
            **kwargs: ``use_entmax`` controls teacher probability computation.

        Returns:
            Dictionary with teacher probabilities, softmax outputs, logits and
            hard predictions stacked across teachers.
        """
        use_entmax = kwargs.get('use_entmax', True)

        all_teacher_logits = []
        teachers_probabilities_list = []
        teachers_predictions_list = []
        teachers_softmax_list = []
        teachers_logits_list = []
        if teacher_model_dict_list is not None and len(teacher_model_dict_list) > 0:
            if use_training:
                loader = self.data_module.train_loader_eval(batch_size=None)
            else:
                loader = self.data_module.val_loader(batch_size=512)

            for teacher_model_dict in teacher_model_dict_list:
                teacher_model = copy.deepcopy(self.model)
                teacher_model.load_state_dict(
                    copy.deepcopy(teacher_model_dict))
                teacher_model.to(self.device)
                teacher_model.eval()

                teacher_outputs_list = []
                for batch in loader:
                    inputs = batch['data'].float().to(self.device)
                    with torch.no_grad():
                        outputs = teacher_model(inputs)
                    teacher_outputs_list.append(outputs)
                    all_teacher_logits.append(outputs)

                if len(teacher_outputs_list) > 0:
                    stacked_outputs = torch.cat(teacher_outputs_list, dim=0)
                else:
                    stacked_outputs = torch.tensor([], device=self.device)

                if stacked_outputs.numel() > 0:
                    teacher_softmax = torch.nn.functional.softmax(
                        stacked_outputs / 1.0, dim=-1)
                    teacher_predictions = torch.argmax(stacked_outputs, dim=-1)

                    if use_entmax:
                        teacher_probabilities = entmax_bisect(
                            stacked_outputs * 10, alpha=1.5, dim=-1)
                    else:
                        teacher_probabilities = torch.nn.functional.one_hot(
                            teacher_predictions, num_classes=stacked_outputs.size(
                                -1)
                        ).float()

                    if torch.isnan(teacher_probabilities).any():
                        raise ValueError("Teacher Probabilities contiene NaN!")

                    teachers_probabilities_list.append(teacher_probabilities)
                    teachers_predictions_list.append(teacher_predictions)
                    teachers_softmax_list.append(teacher_softmax)
                    teachers_logits_list.append(stacked_outputs)

        if len(teachers_probabilities_list) > 0:
            teacher_probabilities = torch.stack(
                teachers_probabilities_list, dim=0)
            teacher_predictions = torch.stack(teachers_predictions_list, dim=0)
            teacher_softmax = torch.stack(teachers_softmax_list, dim=0)
            teacher_logits = torch.stack(teachers_logits_list, dim=0)
        else:
            teacher_probabilities = torch.tensor([], device=self.device)
            teacher_predictions = torch.tensor([], device=self.device)
            teacher_softmax = torch.tensor([], device=self.device)
            teacher_logits = torch.tensor([], device=self.device)

        return {
            'teacher_probabilities': teacher_probabilities,
            'teacher_softmax_list': teacher_softmax,
            'teacher_logits_list': teacher_logits,
            'teacher_predictions_list': teacher_predictions,
        }
