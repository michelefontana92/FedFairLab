from dataloaders.base_loader import BaseDataLoader
import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from .datasets import DatasetFactory


class IndexedDataset(torch.utils.data.Dataset):
    """Index-based dataset view that remaps sample indices within each split."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        index_tensor = torch.as_tensor(self.indices, dtype=torch.long)
        self.x = dataset.x[index_tensor]
        self.y = dataset.y[index_tensor]
        self.positive_mask = dataset.positive_mask[index_tensor]
        self.groups = {
            name: values[self.indices]
            for name, values in dataset.groups.items()
        }
        self.groups_tensor = {
            name: values[index_tensor]
            for name, values in dataset.groups_tensor.items()
        }
        self.group_ids = dataset.group_ids
        self.sensitive_attributes = dataset.sensitive_attributes
        self.use_local_weights = dataset.use_local_weights
        self.local_weights = {
            name: values[index_tensor]
            for name, values in dataset.local_weights.items()
        }
        counts = torch.bincount(self.y)
        nonzero = counts > 0
        self.class_weights = torch.zeros(
            len(counts), dtype=torch.float32)
        self.class_weights[nonzero] = (
            len(self.y) / (nonzero.sum() * counts[nonzero].float())
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        result = dict(self.dataset[int(self.indices[index])])
        # Teacher caches are split-local, so indices must also be split-local.
        result['index'] = index
        result['class_weights'] = self.class_weights
        return result

    def get_class_weights(self):
        return self.class_weights

    def get_group_ids(self):
        return self.group_ids

    def get_group_cardinality(self, y, group_id, training_group_name):
        return len(torch.where(
            (self.groups_tensor[training_group_name] == group_id)
            & (self.y == y)
        )[0])

class DataModule(BaseDataLoader):
    """
    DataModule class for loading and managing datasets.
    Attributes:
        kwargs (dict): Keyword arguments for configuring the DataModule.
        dataset_name (str): Name of the dataset.
        root (str): Root directory for the dataset.
        train_set_name (str): Name of the training set.
        val_set_name (str): Name of the validation set.
        test_set_name (str): Name of the test set.
        batch_size (int): Batch size for data loading. Default is 128.
        num_workers (int): Number of workers for data loading. Default is 0.
        load_test_set (bool): Flag to indicate whether to load the test set. Default is False.
        datasets (dict): Dictionary containing the datasets.
    Methods:
        _load_data(): Loads the datasets based on the provided paths.
        train_loader(batch_size=None): Returns a DataLoader for the training set.
        val_loader(batch_size=None): Returns a DataLoader for the validation set.
        test_loader(batch_size=None): Returns a DataLoader for the test set.
        train_loader_eval(batch_size=None): Returns a DataLoader for evaluating the training set.
        get_input_dim(): Returns the input dimension of the training set.
        get_class_weights(): Returns the class weights of the training set.
        merge(datamodule_list): Merges the datasets from a list of DataModules.
        get_group_ids(): Returns the group IDs of the training set.
        get_group_cardinality(y, group_id, training_group_name): Returns the group cardinality of the training set.
        serialize(): Serializes the DataModule.
        deserialize(data): Reconstructs an instance of DataModule from serialized data.
    """
    def __init__(self, **kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        self.kwargs = kwargs
        
        self.dataset_name = kwargs.get('dataset')
        self.root = kwargs.get('root')
        self.train_set_name = kwargs.get('train_set')
        self.val_set_name = kwargs.get('val_set')
        self.test_set_name = kwargs.get('test_set')
        
       
        self.batch_size = kwargs.get('batch_size', 128)
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory', torch.cuda.is_available())
        self.load_test_set = kwargs.get('load_test_set', False)
        self.validation_strategy = kwargs.get('validation_strategy', 'external')
        self.validation_fraction = float(kwargs.get('validation_fraction', 0.2))
        self.split_seed = int(kwargs.get('split_seed', 42))
        self.stratify_columns = tuple(kwargs.get('stratify_columns') or ())
        self.holdout_indices_path = kwargs.get('holdout_indices_path')
        self.cv_folds = int(kwargs.get('cv_folds', 5))
        self.fold_id = kwargs.get('fold_id')
        if self.fold_id is not None:
            self.fold_id = int(self.fold_id)
        self._dataset_kwargs = self.kwargs
        self._load_data()
    
    def _load_data(self):
        """Handle load data."""
        self.datasets = {
            'train': None,
            'val': None,
        }

        # Keep the test split completely untouched during training. It is
        # instantiated lazily by test_loader() only after model selection.
        if self.load_test_set:
            self.datasets['test'] = None
            
        if self.validation_strategy == 'holdout':
            self._load_holdout_data()
        elif self.validation_strategy == 'kfold':
            self._load_kfold_data()
        elif self.validation_strategy == 'external':
            for key, filename in {
                    'train': self.train_set_name,
                    'val': self.val_set_name}.items():
                self.datasets[key] = DatasetFactory().create_dataset(
                    filename=filename,
                    **self.kwargs
                )
        else:
            raise ValueError(
                f"Unsupported validation strategy: {self.validation_strategy}")

    def _load_holdout_data(self):
        """Create persistent stratified train/validation views of the train pool."""
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not self.stratify_columns:
            raise ValueError(
                "stratify_columns are required for holdout validation")

        source_path = os.path.join(self.root, self.train_set_name)
        source_hash = self._file_sha256(source_path)
        indices_path = self.holdout_indices_path or self._default_indices_path(
            source_path, 'holdout')
        raw_data = pd.read_csv(source_path)
        missing_columns = [
            column for column in self.stratify_columns
            if column not in raw_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing holdout stratification columns: {missing_columns}")

        if os.path.exists(indices_path):
            train_indices, val_indices = self._load_holdout_indices(
                indices_path, source_hash, len(raw_data))
        else:
            strata = self._build_strata(raw_data, minimum_count=2)
            all_indices = np.arange(len(raw_data), dtype=np.int64)
            train_indices, val_indices = train_test_split(
                all_indices,
                test_size=self.validation_fraction,
                random_state=self.split_seed,
                shuffle=True,
                stratify=strata,
            )
            train_indices = np.sort(train_indices)
            val_indices = np.sort(val_indices)
            os.makedirs(os.path.dirname(indices_path) or '.', exist_ok=True)
            np.savez_compressed(
                indices_path,
                train_indices=train_indices,
                val_indices=val_indices,
                source_sha256=np.asarray(source_hash),
                source_rows=np.asarray(len(raw_data)),
                validation_fraction=np.asarray(self.validation_fraction),
                split_seed=np.asarray(self.split_seed),
                stratify_columns=np.asarray(self.stratify_columns),
            )

        self._build_indexed_datasets(
            source_path,
            source_hash,
            train_indices,
            val_indices,
            split_name='holdout',
        )

    def _load_kfold_data(self):
        """Load one persistent jointly stratified K-fold partition."""
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.fold_id is None or not 0 <= self.fold_id < self.cv_folds:
            raise ValueError(
                f"fold_id must be between 0 and {self.cv_folds - 1}")
        if not self.stratify_columns:
            raise ValueError(
                "stratify_columns are required for K-fold validation")

        source_path = os.path.join(self.root, self.train_set_name)
        source_hash = self._file_sha256(source_path)
        indices_path = self.holdout_indices_path or self._default_indices_path(
            source_path, f'kfold_{self.cv_folds}')
        raw_data = pd.read_csv(source_path)
        missing_columns = [
            column for column in self.stratify_columns
            if column not in raw_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing K-fold stratification columns: {missing_columns}")

        if os.path.exists(indices_path):
            assignments = self._load_kfold_assignments(
                indices_path, source_hash, len(raw_data))
        else:
            strata = self._build_strata(
                raw_data, minimum_count=self.cv_folds)
            assignments = np.full(len(raw_data), -1, dtype=np.int16)
            splitter = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.split_seed,
            )
            all_indices = np.arange(len(raw_data), dtype=np.int64)
            for fold, (_, val_indices) in enumerate(
                    splitter.split(all_indices, strata)):
                assignments[val_indices] = fold
            if np.any(assignments < 0):
                raise RuntimeError("K-fold assignment did not cover every row")
            os.makedirs(os.path.dirname(indices_path) or '.', exist_ok=True)
            np.savez_compressed(
                indices_path,
                fold_assignments=assignments,
                source_sha256=np.asarray(source_hash),
                source_rows=np.asarray(len(raw_data)),
                cv_folds=np.asarray(self.cv_folds),
                split_seed=np.asarray(self.split_seed),
                stratify_columns=np.asarray(self.stratify_columns),
            )

        val_indices = np.flatnonzero(assignments == self.fold_id)
        train_indices = np.flatnonzero(assignments != self.fold_id)
        self._validate_partition(
            train_indices, val_indices, len(raw_data), indices_path)
        self._build_indexed_datasets(
            source_path,
            source_hash,
            train_indices,
            val_indices,
            split_name=f'kfold_{self.cv_folds}_fold_{self.fold_id}',
        )

    def _build_indexed_datasets(self, source_path, source_hash,
                                train_indices, val_indices, split_name):
        """Instantiate the pool dataset and expose indexed train/val views.

        Preprocessing keeps the dataset wrapper's configured clean reference
        and scaler. Only the samples exposed to the training and validation
        loaders are selected here.
        """
        dataset_kwargs = dict(self.kwargs)
        self._dataset_kwargs = dataset_kwargs
        full_dataset = DatasetFactory().create_dataset(
            filename=self.train_set_name,
            **dataset_kwargs
        )
        self.datasets['train'] = IndexedDataset(full_dataset, train_indices)
        self.datasets['val'] = IndexedDataset(full_dataset, val_indices)

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _default_indices_path(self, source_path, split_name):
        """Return a compact split path unique to its stratification settings."""
        split_configuration = '|'.join((
            split_name,
            str(self.split_seed),
            str(self.validation_fraction),
            *self.stratify_columns,
        ))
        split_id = hashlib.sha256(
            split_configuration.encode('utf-8')).hexdigest()[:10]
        return os.path.join(
            os.path.dirname(source_path),
            f'{self.dataset_name}_{split_name}_{split_id}_indices.npz',
        )

    def _build_strata(self, raw_data, minimum_count):
        """Build joint strata, pooling only undersized cells by target label.

        Rare intersectional cells cannot be split independently. Pooling them
        by the first stratification column preserves the target distribution
        while retaining exact joint stratification for sufficiently populated
        cells.
        """
        selected = raw_data[list(self.stratify_columns)].astype(str)
        strata = selected.agg('\x1f'.join, axis=1)
        counts = strata.value_counts()
        rare_mask = strata.map(counts).lt(minimum_count)
        if rare_mask.any():
            target_values = selected.iloc[:, 0]
            strata = strata.mask(
                rare_mask,
                '__rare__\x1f' + target_values,
            )

        pooled_counts = strata.value_counts()
        if pooled_counts.min() < minimum_count:
            affected = int((pooled_counts < minimum_count).sum())
            raise ValueError(
                "Joint stratification remains too sparse after pooling rare "
                f"cells by target label. Affected strata: {affected}")
        return strata

    def _load_holdout_indices(self, path, source_hash, source_rows):
        with np.load(path, allow_pickle=False) as saved:
            expected = {
                'source_sha256': source_hash,
                'source_rows': source_rows,
                'validation_fraction': self.validation_fraction,
                'split_seed': self.split_seed,
            }
            actual = {
                key: saved[key].item()
                for key in expected
            }
            actual_columns = tuple(saved['stratify_columns'].tolist())
            if actual != expected or actual_columns != self.stratify_columns:
                raise ValueError(
                    f"Stored holdout metadata does not match current data/config: {path}")
            train_indices = saved['train_indices'].astype(np.int64)
            val_indices = saved['val_indices'].astype(np.int64)

        self._validate_partition(
            train_indices, val_indices, source_rows, path)
        return train_indices, val_indices

    def _load_kfold_assignments(self, path, source_hash, source_rows):
        with np.load(path, allow_pickle=False) as saved:
            expected = {
                'source_sha256': source_hash,
                'source_rows': source_rows,
                'cv_folds': self.cv_folds,
                'split_seed': self.split_seed,
            }
            actual = {key: saved[key].item() for key in expected}
            actual_columns = tuple(saved['stratify_columns'].tolist())
            if actual != expected or actual_columns != self.stratify_columns:
                raise ValueError(
                    f"Stored K-fold metadata does not match current data/config: {path}")
            assignments = saved['fold_assignments'].astype(np.int64)
        if len(assignments) != source_rows:
            raise ValueError(
                f"Stored K-fold assignments have the wrong length: {path}")
        if np.any((assignments < 0) | (assignments >= self.cv_folds)):
            raise ValueError(
                f"Stored K-fold assignments contain invalid fold ids: {path}")
        return assignments

    @staticmethod
    def _validate_partition(train_indices, val_indices, source_rows, path):
        if np.intersect1d(train_indices, val_indices).size:
            raise ValueError(f"Stored train/validation indices overlap: {path}")
        combined = np.sort(np.concatenate([train_indices, val_indices]))
        if not np.array_equal(combined, np.arange(source_rows)):
            raise ValueError(
                f"Stored indices do not partition the source rows: {path}")
       

    def train_loader(self,batch_size=None):
        """Return the training data loader.
        
        Args:
            batch_size: Number of samples per batch; None keeps the loader default.
        
        Returns:
            Requested result.
        """
        return DataLoader(self.datasets.get('train'),
                          batch_size=self.batch_size if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory,
                          #persistent_workers=True
                          )

    def val_loader(self,batch_size=None):
       """Return the validation data loader.
       
       Args:
           batch_size: Number of samples per batch; None keeps the loader default.
       
       Returns:
           Requested result.
       """
       return DataLoader(self.datasets.get('val'),
                          batch_size=len(self.datasets.get('val')),#if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=self.pin_memory,
                          #persistent_workers=True
                          )

    def test_loader(self,batch_size=None):
        """Return the test data loader.
        
        Args:
            batch_size: Number of samples per batch; None keeps the loader default.
        
        Returns:
            Requested result.
        """
        if not self.load_test_set:
            raise RuntimeError(
                "Test set is not loaded. Initialize DataModule with "
                "load_test_set=True and a valid test_set path."
            )
        test_dataset = self.datasets.get('test')
        if test_dataset is None:
            test_dataset = DatasetFactory().create_dataset(
                filename=self.test_set_name,
                **self._dataset_kwargs
            )
            self.datasets['test'] = test_dataset
        return DataLoader(test_dataset,
                          batch_size=len(test_dataset) if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=self.pin_memory,
                          #persistent_workers=True
                         )

    def train_loader_eval(self,batch_size=None):
        """Return the evaluation training data loader.
        
        Args:
            batch_size: Number of samples per batch; None keeps the loader default.
        
        Returns:
            Requested result.
        """
        return DataLoader(self.datasets.get('train'),
                          batch_size=len(self.datasets.get('train')), #if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=self.pin_memory,
                          #persistent_workers=True
                          )
    
    def get_input_dim(self):
        """Return input dim.
        
        Returns:
            Requested result.
        """
        return self.datasets['train'].x.shape[1]
    
    def get_class_weights(self):
        """Return class weights.
        
        Returns:
            Requested result.
        """
        return self.datasets['train'].get_class_weights()
    
    def merge(self, datamodule_list):
        """Merge.
        
        Args:
            datamodule_list: Data modules to merge.
        """
        for key in self.datasets.keys():
            for datamodule in datamodule_list:
                self.datasets[key].merge(datamodule.datasets[key])
        return self
    
    def get_group_ids(self):
        """Return group ids.
        
        Returns:
            Requested result.
        """
        return self.datasets['train'].get_group_ids()
    
    def get_group_cardinality(self,y,group_id,training_group_name):
        """Return group cardinality.
        
        Args:
            y: Label tensor or array.
            group_id: Group identifier used for filtering or statistics.
            training_group_name: Name of the sensitive/group attribute.
        
        Returns:
            Requested result.
        """
        return self.datasets['train'].get_group_cardinality(y,group_id,training_group_name)
    
    def serialize(self):
        """
        Serializza il DataModule.
        """
        if self.validation_strategy in {'holdout', 'kfold'}:
            return {'kwargs': self.kwargs, 'datasets': None}
        return {
            'kwargs': self.kwargs,  # Argomenti usati per creare il DataModule
            'datasets': {
                key: DatasetFactory.serialize(dataset) if dataset else None
                for key, dataset in self.datasets.items()
            }
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza di DataModule dai dati serializzati.
        """
        instance = DataModule(**data['kwargs'])
        if data['datasets'] is None:
            return instance
        instance.datasets = {
            key: DatasetFactory.deserialize(dataset_data) if dataset_data else None
            for key, dataset_data in data['datasets'].items()
        }
        return instance
