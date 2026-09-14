from .surrogate_factory import register_surrogate
import torch


class _BaseDistillationSurrogate:
    """
    Shared utilities for FedFairLab distillation surrogates.

    The server-side aggregation phase now sends each client a single
    precomputed ensemble target ``z_ens``. Local learners expose that target as
    ``teacher_logits_list`` with shape ``[1, batch_size, num_classes]``.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize distillation hyperparameters.

        Args:
            **kwargs: Optional ``temperature`` and ``lambda_global`` values.
        """
        self.name = kwargs.get('name', 'fedfairlab_distillation')
        self.weight = kwargs.get('weight', 1.0)
        self.temperature = float(kwargs.get('temperature', 2.0))
        self.lambda_global = kwargs.get('lambda_global', 1.0)
        self.xi = kwargs.get('xi', 0.8)
        self.weights_list = []

    def set_weights(self, weights):
        """
        Keep compatibility with callers that set aggregation weights.

        The faithful FedFairLab aggregation uses the server-computed ``z_ens``
        directly, so no local re-weighting is applied in these surrogates.

        Args:
            weights: Ignored compatibility value.
        """
        self.weights_list = weights

    def _teacher_logits(self, teacher_logits_list):
        """
        Return the ensemble target logits sent by the server.

        Args:
            teacher_logits_list: Tensor/list shaped ``[T, B, C]`` or ``[B, C]``.

        Returns:
            Tensor shaped ``[B, C]``.
        """
        assert teacher_logits_list is not None, 'teacher_logits_list must be provided'
        if isinstance(teacher_logits_list, torch.Tensor):
            if teacher_logits_list.numel() == 0:
                return None
            if teacher_logits_list.dim() == 2:
                return teacher_logits_list
            return teacher_logits_list.mean(dim=0)
        if len(teacher_logits_list) == 0:
            return None
        return torch.stack(teacher_logits_list, dim=0).mean(dim=0)

    def _kl_per_sample(self, logits, teacher_logits):
        """
        Compute per-sample KL distillation loss against teacher logits.

        Args:
            logits: Student logits with shape ``[B, C]``.
            teacher_logits: Target logits with shape ``[B, C]``.

        Returns:
            Tensor shaped ``[B]``.
        """
        temperature = self.temperature
        student_log_probs = torch.log_softmax(logits / temperature, dim=1)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=1).detach()
        kl = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none',
            log_target=False,
        ).sum(dim=1)
        return kl * (temperature ** 2)

    def _wasserstein_per_sample(self, student_distribution, teacher_distribution):
        """
        Compute W1 distance per sample along the ordered class axis.

        Args:
            student_distribution: Student probabilities with shape ``[B, C]``.
            teacher_distribution: Teacher probabilities with shape ``[B, C]``.

        Returns:
            Tensor shaped ``[B]``.
        """
        student_cdf = torch.cumsum(student_distribution, dim=1)
        teacher_cdf = torch.cumsum(teacher_distribution, dim=1)
        return torch.abs(student_cdf - teacher_cdf).sum(dim=1)

    def _selective_mask(self, probabilities, teacher_logits, labels):
        """
        Build FedFairLAB's selective distillation mask.

        The mask keeps examples where the teacher is correct and the student is
        currently wrong.

        Args:
            probabilities: Student Entmax probabilities.
            teacher_logits: Teacher logits.
            labels: Ground-truth labels.

        Returns:
            Float tensor shaped ``[B]``.
        """
        with torch.no_grad():
            student_pred = torch.argmax(probabilities, dim=1)
            teacher_pred = torch.argmax(teacher_logits, dim=1)
            labels = labels.long().view(-1)
            return ((teacher_pred == labels) & (student_pred != labels)).float()

    def _selective_wasserstein_loss(
            self, logits, probabilities, teacher_logits, labels):
        """
        Compute masked W1 distillation loss against a teacher.

        Args:
            logits: Student logits used to build its softened distribution.
            probabilities: Student Entmax probabilities used only by the mask.
            teacher_logits: Teacher logits.
            labels: Ground-truth labels.

        Returns:
            Scalar tensor. Returns zero when no example satisfies the mask.
        """
        student_distribution = torch.softmax(
            logits / self.temperature, dim=1)
        teacher_distribution = torch.softmax(
            teacher_logits / self.temperature, dim=1).detach()
        mask = self._selective_mask(probabilities, teacher_logits, labels)
        distances = self._wasserstein_per_sample(
            student_distribution, teacher_distribution)
        return (distances * mask).sum() / mask.sum().clamp(min=1.0)


@register_surrogate('fedfairlab_ensemble_distillation')
class FedFairLabEnsembleDistillation(_BaseDistillationSurrogate):
    """
    Minimized objective for the FedFairLab aggregation/distillation phase.

    The loss is the KL divergence between the client model and the server
    ensemble target ``z_ens``. If no teacher target is available, the loss falls
    back to supervised cross-entropy on the client's labels.
    """

    def __call__(self, **kwargs):
        """
        Compute scalar aggregation loss.

        Args:
            **kwargs: Requires ``logits``, ``labels`` and ``teacher_logits_list``.

        Returns:
            Scalar tensor minimized by the local learner.
        """
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        teacher_logits = self._teacher_logits(kwargs.get('teacher_logits_list'))

        assert logits is not None, 'logits must be provided'
        if torch.isnan(logits).any():
            raise ValueError('Student logits contain NaN')

        if teacher_logits is None:
            assert labels is not None, 'labels must be provided when no teacher is available'
            return torch.nn.functional.cross_entropy(logits, labels.long().view(-1,))
        if torch.isnan(teacher_logits).any():
            raise ValueError('Teacher logits contain NaN')
        assert teacher_logits.shape == logits.shape, 'Teacher and student logits must match'

        return self._kl_per_sample(logits, teacher_logits).mean()


@register_surrogate('fedfairlab_ensemble_distillation_batch')
class FedFairLabEnsembleDistillationBatch(_BaseDistillationSurrogate):
    """
    Per-sample version of the aggregation distillation loss.

    This is used where the local learner expects a batch-shaped objective for
    optional group-aware weighting.
    """

    def __call__(self, **kwargs):
        """
        Compute per-sample aggregation losses.

        Args:
            **kwargs: Requires ``logits``, ``labels`` and ``teacher_logits_list``.

        Returns:
            Tensor shaped ``[batch_size]``.
        """
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        teacher_logits = self._teacher_logits(kwargs.get('teacher_logits_list'))

        assert logits is not None, 'logits must be provided'
        if teacher_logits is None:
            assert labels is not None, 'labels must be provided when no teacher is available'
            return torch.nn.functional.cross_entropy(
                logits,
                labels.long().view(-1,),
                reduction='none',
            )
        assert teacher_logits.shape == logits.shape, 'Teacher and student logits must match'
        return self._kl_per_sample(logits, teacher_logits)


@register_surrogate('fedfairlab_ensemble_distillation_score')
class FedFairLabEnsembleDistillationScore(FedFairLabEnsembleDistillation):
    """
    Maximized score counterpart of the aggregation distillation objective.

    Checkpoint selection maximizes ``val_constraints_score``. Returning the
    negative KL therefore selects the model closest to ``z_ens``.
    """

    def __call__(self, **kwargs):
        """Return the negative scalar distillation loss."""
        return -super().__call__(**kwargs)


@register_surrogate('fedfairlab_local_adaptive_objective')
class FedFairLabLocalAdaptiveObjective(FedFairLabEnsembleDistillation):
    """
    Local adaptive objective used when the server builds the constrained problem.

    It implements Eq. 3 of FedFairLAB: supervised task loss plus selective
    Wasserstein distillation from the sampled global teacher.
    """

    def __call__(self, **kwargs):
        """
        Compute local adaptive training loss.

        Args:
            **kwargs: Requires ``logits`` and ``labels``; teacher logits are
                optional.

        Returns:
            Scalar tensor minimized by ALM.
        """
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits = self._teacher_logits(kwargs.get('teacher_logits_list'))

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        ce_loss = torch.nn.functional.cross_entropy(logits, labels.long().view(-1,))
        if teacher_logits is None:
            return ce_loss
        assert teacher_logits.shape == logits.shape, 'Teacher and student logits must match'
        distill_loss = self._selective_wasserstein_loss(
            logits,
            probabilities,
            teacher_logits,
            labels,
        )
        return self.xi * ce_loss + (1.0 - self.xi) * self.lambda_global * distill_loss


@register_surrogate('fedfairlab_local_adaptive_batch_objective')
class FedFairLabLocalAdaptiveBatchObjective(FedFairLabEnsembleDistillationBatch):
    """
    Per-sample local adaptive objective for optional group-aware weighting.
    """

    def __call__(self, **kwargs):
        """
        Compute per-sample local adaptive losses.

        Args:
            **kwargs: Requires ``logits`` and ``labels``; teacher logits are
                optional.

        Returns:
            Tensor shaped ``[batch_size]``.
        """
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits = self._teacher_logits(kwargs.get('teacher_logits_list'))

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            labels.long().view(-1,),
            reduction='none',
        )
        if teacher_logits is None:
            return ce_loss
        assert teacher_logits.shape == logits.shape, 'Teacher and student logits must match'
        student_distribution = torch.softmax(
            logits / self.temperature, dim=1)
        teacher_distribution = torch.softmax(
            teacher_logits / self.temperature, dim=1).detach()
        mask = self._selective_mask(probabilities, teacher_logits, labels)
        distill_loss = self._wasserstein_per_sample(
            student_distribution, teacher_distribution) * mask
        return self.xi * ce_loss + (1.0 - self.xi) * self.lambda_global * distill_loss

