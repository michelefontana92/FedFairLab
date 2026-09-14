from abc import ABC, abstractmethod
import os
import shutil

from debug_utils import debug_print


VALIDATION_SPLIT_SEED = 42


class BaseRun(ABC):
    """Implementation of BaseRun."""
    def __init__(self,**kwargs):
        
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        self.model = kwargs.get('model')
        self.dataset = kwargs.get('dataset')
        self.data_file_prefix = kwargs.get('data_file_prefix')
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.project_name = kwargs.get('project_name')
        self.data_root = kwargs.get('data_root')
        self.keep_checkpoints = kwargs.get('keep_checkpoints', True)
        self.evaluate_only = kwargs.get('evaluate_only', False)
        self.evaluate_test = kwargs.get('evaluate_test', False)
        self.num_workers = 0
        self.aggregation_patience = kwargs.get('aggregation_patience', 0)
        self.aggregation_min_delta = kwargs.get('aggregation_min_delta', 1e-6)
        self.validation_strategy = kwargs.get('validation_strategy', 'external')
        self.validation_fraction = kwargs.get('validation_fraction', 0.2)
        self.split_seed = VALIDATION_SPLIT_SEED
        self.random_seed = kwargs.get('random_seed')
        self.global_patience = kwargs.get('global_patience', 5)
        self.stratify_columns = tuple(kwargs.get('stratify_columns') or ())
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.fold_id = kwargs.get('fold_id')

    @staticmethod
    def resolve_data_root(kwargs, folder_name, *fallback_folder_names):
        """Return the repository-local directory for a built-in dataset.

        ``--data_root`` denotes the common directory containing all dataset
        folders. Fallback names preserve compatibility with older layouts.
        """
        repository_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..'))
        common_data_root = kwargs.get('data_root') or os.path.join(
            repository_root, 'data')
        candidates = [
            os.path.join(common_data_root, name)
            for name in (folder_name, *fallback_folder_names)
        ]
        return next(
            (path for path in candidates if os.path.isdir(path)),
            candidates[0],
        )

    @classmethod
    def resolve_experiment_data_root(cls, kwargs, dataset_folder):
        """Return ``<data_root>/<experiment_name>/<dataset_folder>``."""
        experiment_name = kwargs.get('experiment_name') or '10_Clients'
        return cls.resolve_data_root(
            kwargs,
            os.path.join(experiment_name, dataset_folder),
            dataset_folder,
        )

    def configure_validation_splits(
            self, kwargs, target_column, sensitive_columns=()):
        """Configure holdout/K-fold validation without accessing test data.

        Unless columns are supplied explicitly, stratification uses the target
        jointly with the paper's sensitive columns for this dataset.
        """
        self.validation_strategy = kwargs.get(
            'validation_strategy', 'holdout')
        self.validation_fraction = kwargs.get('validation_fraction', 0.2)
        self.split_seed = VALIDATION_SPLIT_SEED
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.fold_id = kwargs.get('fold_id')

        explicit_columns = tuple(kwargs.get('stratify_columns') or ())
        if explicit_columns:
            self.stratify_columns = explicit_columns
            return

        self.stratify_columns = tuple(dict.fromkeys(
            (target_column, *sensitive_columns)))

    def compute_group_cardinality(self,group_name):
        """Compute group cardinality.
        
        Args:
            group_name: Name of the sensitive/group attribute.
        
        Returns:
            Requested result.
        """
        for name,group_dict in self.sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    @abstractmethod
    def setUp(self):
        """Handle setUp."""
        pass 
    
    def tearDown(self):
        """
        Clean local checkpoint artifacts after a completed training run.

        Checkpoints are kept by default so the run can be inspected or
        re-evaluated locally. When ``keep_checkpoints`` is false, the
        project-specific checkpoint directory is removed after final metrics and
        WandB artifacts have already been logged.
        """
        if self.keep_checkpoints or self.evaluate_only:
            return

        checkpoint_dir = self._checkpoint_dir()
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            debug_print(f"Removed checkpoint directory: {checkpoint_dir}")

    def _checkpoint_dir(self):
        """
        Return the checkpoint directory used by this run.

        Returns:
            Project-specific checkpoint directory.
        """
        if hasattr(self, "builder"):
            return self.builder.common_client_params.get(
                "checkpoint_dir", f"checkpoints/{self.project_name}"
            )
        return f"checkpoints/{self.project_name}"

    @abstractmethod
    def run(self,**kwargs):
        """Handle run.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
    
    def __call__(self, **kwargs):
        """Evaluate the callable object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        self.setUp()
        self.run(**kwargs)
        self.tearDown()

    def build_server_config(self,**kwargs):
        """Handle build server config.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        server_config = {
            'early_stopping_patience': self.global_patience,
            'monitor':'global_val_requirements',
            'mode':'min',
        }
        return server_config
    

    def to_dict(self):
        """Handle to dict."""
        return {
            'model': self.model,
            'dataset': self.dataset,
            'data_file_prefix': self.data_file_prefix or self.dataset,
            'sensitive_attributes': self.sensitive_attributes,
            'project_name': self.project_name,
            'data_root': self.data_root,
            'clean_data_path': getattr(self, 'clean_data_path', None),
            'validation_strategy': self.validation_strategy,
            'validation_fraction': self.validation_fraction,
            'split_seed': self.split_seed,
            'random_seed': self.random_seed,
            'global_patience': self.global_patience,
            'stratify_columns': self.stratify_columns,
            'cv_folds': self.cv_folds,
            'fold_id': self.fold_id,
            'evaluate_test': self.evaluate_test,
            'num_workers': self.num_workers,
            'aggregation_patience': self.aggregation_patience,
            'aggregation_min_delta': self.aggregation_min_delta,
            'lr': self.learning_rate,
            'batch_size': self.batch_size,
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'input': self.input,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'output': self.output,
            'server_config': self.build_server_config(),
        }
