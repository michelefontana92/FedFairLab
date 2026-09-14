# FedFairLAB

FedFairLAB is a method for fairness-aware federated learning under
performance budgets. It implements a federated version of the FairLAB
constrained-optimization workflow (see this [paper](link.springer.com/chapter/10.1007/978-3-032-05962-8_13) for a detailed description of FairLAB) and uses ensemble-logit distillation to build
the global model from locally updated client models. Full details about FedFairLAB can be found in the paper 
[Federated learning with multiple, intersectional and multiclass fairness guarantees under performance budgets](https://link.springer.com/article/10.1007/s10618-026-01243-6).

The code is designed for experimental evaluation on horizontally partitioned
tabular datasets, with support for binary and multiclass classification,
multiple sensitive groups, intersectional fairness constraints, and non-IID
client splits.

## Citation

If you use FedFairLAB in your research, please cite the following paper:

```bibtex
@article{DBLP:journals/datamine/FontanaNM26,
  author  = {Michele Fontana and Francesca Naretto and Anna Monreale},
  title   = {Federated learning with multiple, intersectional and multiclass fairness guarantees under performance budgets},
  journal = {Data Min. Knowl. Discov.},
  volume  = {40},
  number  = {5},
  pages   = {81},
  year    = {2026},
  doi     = {10.1007/S10618-026-01243-6}
}
```

## Method Summary

At each federated round, FedFairLAB follows the main phases below.

1. The server samples a teacher model from the bounded global history.
2. Selected clients run a local constrained update using the sampled teacher and
   the selective distillation objective.
3. The server receives the updated local models.
4. The server evaluates the updated models and converts their global scores into
   ensemble weights.
5. Each selected client computes the weighted ensemble logits on its local data.
6. These logits are used as local targets for federated distillation. The loss
   combines a sample/class mean KL with a classwise mean-max KL over the
   observed sensitive groups.
7. Client distillation updates are combined with FedAvg to produce the next
   global model.
8. The new global model is evaluated and inserted into the server history.

The final model is selected from the global checkpointing/history process. The
personalization phase has intentionally been removed from the current
implementation.

## Performance Budget

The performance budget uses the FairLAB semantics:

```text
F1 >= p* - beta
```

where:

- `p*` is the best performance observed so far;
- `beta` is the tolerated performance degradation;
- performance is measured locally for client-side FairLAB updates and globally
  for server-side model scoring.

The performance budget is optional. If `--performance_budget` is omitted, the
performance constraints are not created.

For example, if the best global F1 observed so far is `0.82` and
`--performance_budget 0.05`, the global score penalizes models whose global F1
falls below `0.77`.

The model and `p*` start from scratch. During FairLAB, `p*` can only increase
when validation F1 improves. The improvement constraint
`P >= p* + rho_step` and the budget constraint `P >= p* - beta` belong to the
same performance macro-constraint and are both optimized by the local ALM.
The local model-selection score includes the budget violation but excludes the
improvement violation, so the latter guides optimization without competing
with fairness during checkpoint selection.

```bash
python src/main.py \
  --run compas_fedfairlab \
  --performance_budget 0.05 \
  --performance_step 0.10 \
  -ml demographic_parity -gl Age -tl 0.10
```

## Repository Structure

```text
src/
  main.py                         Command-line entry point.
  builder/                        Experiment builders and Ray actor creation.
  client/                         Federated client actors and client utilities.
  server/                         Federated server logic and aggregation flow.
  wrappers/                       Local learners and FairLAB orchestration.
  surrogates/                     Differentiable objectives and constraints.
  metrics/                        Performance and fairness metrics.
  dataloaders/                    Dataset wrappers and data modules.
  runs/                           Registered experiment configurations.
  architectures/                  Neural network architectures.
  callbacks/                      Checkpointing and early stopping.
  loggers/                        Logging integrations.
```

### Folder Description

- `src/main.py`: command-line entry point. It parses CLI arguments, configures
  visible GPUs, builds the selected run and starts execution.
- `src/builder/`: builders that assemble datasets, models, clients, server,
  callbacks, loggers and Ray resource allocation.
- `src/client/`: Ray client actors. Clients own local data, perform local
  FairLAB updates, evaluate candidate models and compute ensemble logits.
- `src/server/`: server implementations and server-side utilities. The
  FedFairLAB server manages client sampling, global history, aggregation,
  distillation, scoring and checkpointing.
- `src/wrappers/`: training wrappers and orchestration logic. This includes the
  local learner, ALM/FairLAB subproblem selection and the bridge between clients
  and local optimization.
- `src/surrogates/`: differentiable performance/fairness constraints and the
  FedFairLAB adaptive distillation objectives.
- `src/metrics/`: metric implementations used for evaluation and logging.
- `src/dataloaders/`: dataset definitions, preprocessing helpers and data
  modules that expose train/validation/test loaders.
- `src/runs/`: registered experiment configurations. Each run defines model
  dimensions, dataset paths, sensitive attributes and builder setup.
- `src/architectures/`: neural network architectures and the architecture
  factory.
- `src/callbacks/`: early stopping and model checkpoint callbacks.
- `src/loggers/`: logging interfaces and the Weights & Biases integration.
- `src/scoring_utils.py`: shared global/local scoring utilities used by both
  client and server code.

## Installation

Create and activate a Python environment, then install the project in editable
mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the package and exposes the `fedfairlab` command-line entry point.
The project dependencies are declared in `pyproject.toml`; `requirements.txt`
contains the same runtime dependency set for environments that still install
dependencies from a requirements file.

```bash
fedfairlab --help
```

The project uses PyTorch, Ray, TorchMetrics, scikit-learn, entmax and Weights &
Biases.

If your environment requires a specific CUDA-enabled PyTorch wheel, install
PyTorch following the official command for your CUDA version before running
`pip install -e .`. The remaining dependencies are installed by the project
metadata.

For dependency-only workflows, for example on clusters that separate dependency
installation from editable package installation, use:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

### CPU and GPU Usage

By default, FedFairLAB runs in CPU mode. If `--gpu_devices` is omitted, the CLI
sets:

```bash
CUDA_VISIBLE_DEVICES=-1
```

Run on CPU:

```bash
fedfairlab \
  --run compas_fedfairlab \
  --num_clients 4
```

Run on one visible GPU:

```bash
fedfairlab \
  --run compas_fedfairlab \
  --num_clients 4 \
  --gpu_devices 0
```

Run on multiple visible GPUs:

```bash
fedfairlab \
  --run compas_fedfairlab \
  --num_clients 8 \
  --gpu_devices 0 \
  --gpu_devices 1
```

The CLI writes `CUDA_VISIBLE_DEVICES` before the run is built. The builder then
initializes Ray with `num_gpus = len(gpu_devices)`. Client actors receive a
fractional GPU share:

```text
num_gpus_per_client = num_visible_gpus / num_clients
```

For example, with `--gpu_devices 0` and `--num_clients 4`, each client actor is
scheduled with `0.25` GPU. With two visible GPUs and eight clients, each client
also receives `0.25` GPU. The FedFairLAB server is local in the current
implementation; GPU resources are assigned to client actors, while model
parameters exchanged through Ray are moved back to CPU before serialization.

### Logging Requirement

At the moment, the codebase is wired to use **Weights & Biases (`wandb`)** as
the experiment logging backend. The server, clients and builder instantiate
`WandbLogger`, and the current training/evaluation flow assumes that this logger
is available.

Before running experiments, make sure W&B is configured:

```bash
wandb login
```

If you want to run without W&B, the clean extension point is `src/loggers/`:
implement a different logger with the same interface and update the
builder/server/client construction points that currently instantiate
`WandbLogger`.

During training, W&B receives the final model artifacts explicitly selected by
checkpointing:

- `global_model`: the best global server checkpoint;
- `<run_id>_client_<n>_local_model`: the final selected local checkpoint for
  client `n`.

Local checkpoint files are kept on disk by default. Add `--delete_checkpoints`
when you want them removed after the completed run has already logged final
metrics and W&B artifacts.

## Core Components

### `FedFairLabBuilder`

`FedFairLabBuilder` lives in `src/builder/fedfairlab_client_builder.py` and is
the main assembly point for experiments. It is responsible for:

- reading the run configuration produced by the selected `Run`;
- assigning CPU/GPU resources for Ray actors;
- creating one `DataModule` per client;
- building the local objectives, fairness constraints and callbacks;
- creating client actors through `ClientFactory`;
- creating the federated server through `ServerFactory`;
- wiring checkpoint directories and W&B logging.

Most experiment-level customization eventually passes through the builder. If
you want to change how clients are created, how resources are allocated, which
logger is used, or which server/client implementation is selected, this is the
first file to inspect.

### `DataModule`

`DataModule` lives in `src/dataloaders/data_loader.py`. It wraps the registered
dataset classes and exposes the loaders consumed by the local learner:

- `train_loader`;
- `train_loader_eval`;
- `val_loader`;
- `test_loader`;
- `get_input_dim`;
- `get_class_weights`;
- `get_group_ids`;
- `get_group_cardinality`.

For each client, the builder creates a `DataModule` with paths derived from:

```text
<data_root>/10_Clients/<dataset_folder>/node_<client_id>/<file_prefix>_train.csv
<data_root>/10_Clients/<dataset_folder>/node_<client_id>/<file_prefix>_test.csv
```

Training, ALM updates, early stopping, checkpoint selection and federated model
selection use only the train and validation splits. Pass `--evaluate_test` to
evaluate the frozen best global checkpoint once on each client’s test split and
log `final_test_*` metrics. Without the flag, training ends with validation
metrics and the test split is never loaded.

With the same flag, each client's best validation-selected local checkpoint is
also evaluated on that client's test split. Explicit local keys use `final_local_val_*` and
`final_local_test_*`; the older `final_val_*` keys are retained for backward
compatibility.

```bash
python src/main.py \
  --run compas_fedfairlab \
  --data_root ./data \
  --validation_strategy holdout \
  --evaluate_test
```

### Persistent holdout validation

Runs can derive validation data from each client's training pool without
copying CSV files. With `validation_strategy=holdout`, the data module creates a
stratified split once and stores only the row indices in a compact
`<dataset>_holdout_<split_id>_indices.npz` file. The identifier distinguishes
different sensitive-group stratifications. The file records the source CSV hash, row
count, fraction, seed and stratification columns; incompatible data or settings
are rejected rather than silently regenerating a different split.

Every built-in run defaults to an 80/20 holdout jointly stratified on the
target and the sensitive attributes used in the paper: Race and Age for
COMPAS, Race and Marital status for the other datasets. The separate
`*_test.csv` is loaded lazily only after checkpoint selection:

```text
compas_train.csv
├── 80% internal train
└── 20% internal validation

compas_test.csv
└── fixed test set
```

The preprocessing scaler keeps the dataset wrapper's configured clean reference;
holdout controls only which rows are exposed by the train and validation
loaders. Use `--validation_fraction` and repeated `--stratify_columns`
arguments to override the defaults. The split seed is fixed to `42` so all
experiments use the same train/validation partition.

Intersectional cells that are too small to be split are pooled by target label.
This preserves exact joint stratification for populated cells and keeps rare
samples usable without reading or modifying the test set.

### Stratified K-fold validation

K-fold validation uses the same training pool and joint stratification without
creating CSV copies. Each client stores one compact
`<dataset>_kfold_<K>_<split_id>_indices.npz` vector. Validation folds are disjoint, every
row appears in validation exactly once. The folds reuse the scaler fitted on
the dataset wrapper's configured clean reference rather than refitting it on
each fold.

Run the five folds as separate reproducible jobs:

```bash
for fold in 0 1 2 3 4; do
  python src/main.py \
    --run compas_fedfairlab \
    --data_root ./data \
    --project_name CompasFairLabCV \
    --id compas_cv \
    --num_clients 10 \
    --validation_strategy kfold \
    --cv_folds 5 \
    --fold_id "$fold"
done
```

The same target and paper-sensitive attributes are used jointly for
stratification unless overridden. Fold checkpoints are isolated under
`checkpoints/<project_name>/fold_<fold_id>/`. CV runs log `cv_val_*` and
`cv_local_val_*` metrics and deliberately do not evaluate the fixed test set.
The test set remains reserved for the validation-selected final training
run.

The `DataModule` delegates actual parsing/preprocessing to the dataset wrapper
registered in `DatasetFactory`.

Client DataLoaders use `num_workers=0` internally because the preprocessed
datasets are already resident in memory and multiprocessing spawn adds
substantial overhead for this workload.

Path resolution follows normal Python process rules: relative dataset paths are
resolved from the directory where you launch `fedfairlab`, not from `src/`.
For reproducible runs, launch commands from the repository root or pass an
absolute `--data_root` pointing to the common directory that contains the
experiment folder, such as `10_Clients`.

## Available Runs

The following experiment identifiers are currently registered:

```text
compas_fedfairlab
education_fedfairlab
employment_fedfairlab
income_fedfairlab
income3_fedfairlab
meps_fedfairlab
```

Use them through `--run`.

## Datasets

The bundled experiment configurations are:

| Run | Dataset folder | CSV prefix | Default joint stratification |
| --- | --- | --- | --- |
| `compas_fedfairlab` | `Compas` | `compas` | `two_year_recid`, `race`, `age_cat` |
| `education_fedfairlab` | `Education` | `education` | `SCHL`, `Race`, `Marital` |
| `employment_fedfairlab` | `Employment` | `employment` | `ESR`, `Race`, `Marital` |
| `income_fedfairlab` | `Income` | `income` | `PINCP`, `Race`, `Marital` |
| `income3_fedfairlab` | `Income_3` | `income_3` | `PINCP`, `Race`, `Marital` |
| `meps_fedfairlab` | `MEPS` | `meps` | `HIGH_EXPENSES`, `RACE`, `MARRY` |

Datasets are implemented in `src/dataloaders/datasets/` and registered through
`@register_dataset("<dataset_key>")`. Each built-in run maps the common
`--data_root` to its dataset folder below `10_Clients`. The builder expects
horizontally split CSV files arranged as:

```text
<data_root>/10_Clients/<dataset_folder>/node_<client_id>/<file_prefix>_train.csv
<data_root>/10_Clients/<dataset_folder>/node_<client_id>/<file_prefix>_test.csv
```

For example, the first COMPAS client is loaded from:

```text
<data_root>/10_Clients/Compas/node_1/compas_train.csv
<data_root>/10_Clients/Compas/node_1/compas_test.csv
```

The built-in runs use the repository's `data` directory by default. If the data
lives elsewhere, pass the common directory containing `10_Clients`:

```bash
fedfairlab \
  --run compas_fedfairlab \
  --data_root /absolute/path/to/data \
  --experiment_name 10_Clients
```

With this command, client 1 is read from:

```text
/absolute/path/to/data/10_Clients/Compas/node_1/compas_train.csv
/absolute/path/to/data/10_Clients/Compas/node_1/compas_test.csv
```

Each dataset wrapper defines:

- the target column;
- categorical columns;
- numerical columns;
- class labels;
- sensitive attributes used for fairness groups;
- optional preprocessing logic.

Each sample returned by a dataset includes features, labels, sensitive-group
ids, positive-label masks, group-id lists and class weights. These fields are
used by the local learner, fairness metrics and differentiable constraints.

If a saved scikit-learn scaler was created with an older scikit-learn version,
FedFairLAB refits it automatically from the dataset clean CSV and overwrites the
stale scaler file. This avoids `InconsistentVersionWarning` messages and keeps
preprocessing tied to the active Python environment.

Scaler fitting always uses the dataset-level clean reference, never a client
train, validation, or test split. For the bundled runs the expected files are:

```text
data/10_Clients/Compas/compas_clean.csv
data/10_Clients/Education/education_clean.csv
data/10_Clients/Employment/employment_clean.csv
data/10_Clients/Income/income_clean.csv
data/10_Clients/Income_3/income_3_clean.csv
data/10_Clients/MEPS/meps_clean.csv
```

The corresponding scaler pickle is created automatically when missing. A run
cannot preprocess a dataset until its clean reference is available.

### Adding a Custom Dataset

To add a new dataset, create a file under `src/dataloaders/datasets/`, for
example `my_dataset.py`:

```python
import os

from .base_dataset import BaseDataset
from .dataset_factory import register_dataset


@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    """Dataset wrapper for a custom tabular classification task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = kwargs.get("root", "data/10_Clients/MyDataset")
        self.data_path = os.path.join(self.root, kwargs["filename"])
        self.scaler_name = kwargs.get("scaler_name", "my_dataset_scalers.p")
        self.scaler_path = os.path.join(self.root, self.scaler_name)
        self.clean_data_path = kwargs.get(
            "clean_data_path",
            os.path.join(self.root, "my_dataset_clean.csv"),
        )

        self.target = "target"
        self.cat_cols = ["gender", "race"]
        self.num_cols = ["age", "income"]
        self.labels = [0, 1]
        self.sensitive_attributes = kwargs.get(
            "sensitive_attributes",
            [
                ("Gender", {"gender": ["Female", "Male"]}),
                ("Race", {"race": ["GroupA", "GroupB", "GroupC"]}),
                (
                    "GenderRace",
                    {
                        "gender": ["Female", "Male"],
                        "race": ["GroupA", "GroupB", "GroupC"],
                    },
                ),
            ],
        )
        self.setup()
```

Then expose the module in `src/dataloaders/datasets/__init__.py` if needed, so
that importing `dataloaders.datasets` registers it.

The CSV files should contain:

- the target column;
- all categorical and numerical feature columns;
- the columns referenced in `sensitive_attributes`.

For multiclass datasets, set `self.labels` to all class labels and configure
the corresponding class count in the run class.

### Adding a Custom Run

A run defines the model shape, dataset name, data paths and sensitive-attribute
metadata. Runs live under `src/runs/` and are registered with
`@register_run("<run_key>")`.

A minimal custom run usually has two files:

```python
# src/runs/MyDataset/my_run.py
import os

from architectures import ArchitectureFactory
from runs.base_run import BaseRun


class MyRun(BaseRun):
    """Base configuration for the custom dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = 10
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.num_classes = 2
        self.output = self.num_classes
        self.model = ArchitectureFactory.create_architecture(
            "mlp2hidden",
            model_params={
                "input": self.input,
                "hidden1": self.hidden1,
                "hidden2": self.hidden2,
                "dropout": self.dropout,
                "output": self.output,
            },
        )
        self.dataset = "my_dataset"
        self.data_file_prefix = "my_dataset"
        self.data_root = self.resolve_experiment_data_root(
            kwargs, "MyDataset"
        )
        self.clean_data_path = kwargs.get(
            "clean_data_path",
            os.path.join(self.data_root, "my_dataset_clean.csv"),
        )
        self.sensitive_attributes = [
            ("Gender", {"gender": ["Female", "Male"]}),
            ("Race", {"race": ["GroupA", "GroupB", "GroupC"]}),
        ]
        self.configure_validation_splits(
            kwargs, "target", ("race", "marital_status")
        )
```

```python
# src/runs/MyDataset/my_fedfairlab.py
from builder import FedFairLabBuilder
from runs.run_factory import register_run

from .my_run import MyRun


@register_run("my_dataset_fedfairlab")
class MyFedFairLABRun(MyRun):
    """FedFairLAB run for the custom dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["run_dict"] = self.to_dict()
        self.builder = FedFairLabBuilder(**kwargs)

    def setUp(self):
        pass

    def run(self):
        self.builder.run()

    def tearDown(self):
        super().tearDown()
```

Finally, import the custom run from `src/runs/__init__.py`, so the decorator is
executed and `RunFactory` can find the run key. You can then launch it with:

```bash
python src/main.py --run my_dataset_fedfairlab --project_name MyFedFairLAB
```

### Adding a Custom Client

Clients are registered through `@register_client("<client_key>")` and created by
`ClientFactory`. The default FedFairLAB client is `client_fedfairlab` in
`src/client/client_fedfairlab.py`.

To create a custom client, subclass or mirror `ClientFedFairLab` and implement
the RPC methods expected by the server:

- `setup`;
- `fit`;
- `compute_weighted_ensemble_logits`;
- `evaluate_constraints`;
- `evaluate_constraints_list`;
- `evaluate_model_from_ckpt`;
- `shutdown`.

Example skeleton:

```python
import ray

from .client_base import BaseClient
from .client_factory import register_client


@register_client("my_client")
@ray.remote(num_cpus=1)
class MyClient(BaseClient):
    """Custom Ray client actor."""

    def __init__(self, **kwargs):
        self.client_name = kwargs.get("client_name")
        self.orchestrator_fn = kwargs.get("orchestrator")

    def setup(self, **kwargs):
        pass

    def fit(self, **kwargs):
        # Return {'params': state_dict, 'weight': fedavg_weight}
        raise NotImplementedError

    def compute_weighted_ensemble_logits(self, **kwargs):
        raise NotImplementedError

    def shutdown(self, **kwargs):
        pass
```

To use a custom client in experiments, update the builder section that calls
`ClientFactory().create(...)` so that it requests `"my_client"` instead of
`"client_fedfairlab"`, or add a new builder branch for your algorithm.

## Customizing Experiments

Most experiment changes fall into one of the following categories.

### 1. Change the Existing CLI Configuration

For standard experiments, prefer CLI options before editing code:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name MyProject \
  --id run_001 \
  --experiment_name 10_Clients \
  --num_clients 4 \
  --clients_per_core 2 \
  --num_federated_iterations 20 \
  --fairlab_orchestrator_epochs 1 \
  --fairlab_local_epochs 5 \
  --history_size 5 \
  --aggregation_epochs 1 \
  --aggregation_local_epochs 2 \
  --performance_budget 0.05 \
  -ml demographic_parity \
  -gl RaceAge \
  -tl 0.20
```

This is the recommended path when changing:

- number of clients;
- FairLAB orchestrator and learner epochs;
- number of server rounds;
- performance budget;
- fairness constraints;
- aggregation/history parameters;
- local learner epochs used during aggregation distillation;
- CPU packing through `--clients_per_core`.

### 2. Add or Modify a Dataset

Use this path when the tabular schema changes. You usually need to:

1. create a dataset wrapper in `src/dataloaders/datasets/`;
2. register it with `@register_dataset("your_dataset_key")`;
3. define target, categorical columns, numerical columns, labels and sensitive
   attributes;
4. expose the module from `src/dataloaders/datasets/__init__.py`;
5. create CSV splits under the expected client directory structure;
6. create or update a run that sets `self.dataset = "your_dataset_key"`.

### 3. Add a New Run

Use this path when the dataset, model dimensions, number of classes or sensitive
attributes change. A run should define:

- `self.input`;
- hidden-layer sizes and dropout;
- `self.num_classes`;
- `self.model`;
- `self.dataset`;
- `self.data_root`;
- `self.sensitive_attributes`.

Then create a registered FedFairLAB run that instantiates `FedFairLabBuilder`.
The run key passed to `@register_run(...)` is the value used by `--run`.

### 4. Change the Model Architecture

Architectures are registered in `src/architectures/`. To add a new architecture:

1. implement a `torch.nn.Module`;
2. register it with `@register_architecture("your_architecture")`;
3. use `ArchitectureFactory.create_architecture(...)` inside your custom run.

### 5. Change Client Behavior

Use this path only when the RPC behavior of clients must change. A custom client
must remain compatible with the server calls:

- local training via `fit`;
- candidate evaluation;
- ensemble-logit computation;
- checkpoint evaluation;
- shutdown.

For most algorithmic changes, it is usually cleaner to modify the orchestrator
or local learner instead of replacing the client actor.

### 6. Change Logging

The current default is W&B through `WandbLogger`. If a different logging backend
is required, implement it in `src/loggers/` and update the builder/server/client
construction points. Until that is done, assume W&B is required for normal
training runs.

## Basic Usage

Show the command-line interface:

```bash
python src/main.py --help
```

Run a small COMPAS experiment with two clients:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas \
  --id compas_debug \
  --experiment_name 10_Clients \
  --num_clients 2 \
  --clients_per_core 2 \
  --num_federated_iterations 2 \
  --fairlab_orchestrator_epochs 1 \
  --fairlab_local_epochs 2 \
  --history_size 2 \
  --aggregation_epochs 1 \
  --aggregation_local_epochs 2
```

Run FedFairLAB without fairness or performance constraints:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas_NoConstraints \
  --id compas_no_constraints \
  --experiment_name 10_Clients \
  --num_clients 2 \
  --clients_per_core 2 \
  --num_federated_iterations 5 \
  --fairlab_orchestrator_epochs 1 \
  --fairlab_local_epochs 5 \
  --history_size 2 \
  --aggregation_epochs 2 \
  --aggregation_local_epochs 3
```

In this configuration no fairness constraints are passed because `-ml`, `-gl`
and `-tl` are omitted. The performance budget is also disabled because
`--performance_budget` is omitted, so the performance constraints are not
created. This is useful as a baseline or for debugging the federated
distillation pipeline.

The epoch-related options in the example control different parts of the
algorithm:

- `--num_federated_iterations 5`: run five server rounds.
- `--fairlab_orchestrator_epochs 1`: run one FairLAB orchestrator step on each
  selected client during the local update phase.
- `--fairlab_local_epochs 5`: train the local learner for five epochs inside
  each FairLAB orchestrator step.
- `--aggregation_epochs 2`: run two server-side aggregation/distillation rounds
  after receiving the locally updated client models.
- `--aggregation_local_epochs 3`: train each client learner for three epochs
  during each aggregation/distillation round.
- `--history_size 2`: keep at most two global models in the server teacher
  history.

Run with one demographic-parity constraint:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas_DP \
  --id compas_dp \
  --experiment_name 10_Clients \
  --num_clients 4 \
  --clients_per_core 2 \
  --num_federated_iterations 10 \
  --fairlab_orchestrator_epochs 1 \
  --fairlab_local_epochs 5 \
  -ml demographic_parity \
  -gl Age \
  -tl 0.10
```

Run with an explicit performance budget:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas_Budget \
  --id compas_budget \
  --experiment_name 10_Clients \
  --num_clients 4 \
  --num_federated_iterations 10 \
  --fairlab_orchestrator_epochs 1 \
  --fairlab_local_epochs 5 \
  --performance_budget 0.05 \
  --performance_step 0.10 \
  -ml demographic_parity \
  -gl Age \
  -tl 0.10
```

Run a multiclass experiment:

```bash
python src/main.py \
  --run education_fedfairlab \
  --project_name FedFairLAB_Education \
  --id education_multiclass \
  --experiment_name 10_Clients \
  --num_clients 4 \
  --num_federated_iterations 10 \
  --fairlab_orchestrator_epochs 1 \
  --fairlab_local_epochs 5 \
  -ml demographic_parity \
  -gl RaceMarital \
  -tl 0.20
```

### Fixed Implementation Settings

The paper configuration keeps the following values outside the CLI:

| Setting | Value | Location |
| --- | --- | --- |
| Learning rate | `1e-4` | Dataset run configuration |
| Training batch size | `128` | Dataset run configuration |
| DataLoader workers | `0` | `BaseRun` |
| Validation split seed | `42` | `BaseRun` |
| Ensemble-weight temperature | `0.05` | FedFairLAB server |
| History-teacher temperature | `0.05` | FedFairLAB server |
| Distillation task weight | `0.8` | FedFairLAB server |
| Distillation temperature | `2.0` | FedFairLAB server |

## CLI Parameters

All experiments are configured through `src/main.py` or the installed
`fedfairlab` command. Parameters used in normal runs are listed first; protocol
and optimization overrides are kept in a separate advanced table.

### Main Parameters

| Option | Default | Description |
| --- | --- | --- |
| `--run`, `-r` | `compas_fedfairlab` | Registered run to execute. Available built-in values include `compas_fedfairlab`, `education_fedfairlab`, `employment_fedfairlab`, `income_fedfairlab`, `income3_fedfairlab`, and `meps_fedfairlab`. |
| `--project_name`, `-p` | `CompasFairLab` | Project namespace used by W&B and by the local checkpoint directory `checkpoints/<project_name>/`. |
| `--id`, `-i` | `test` | Run identifier. It is used in W&B run names and checkpoint artifact names, e.g. `<id>_client_1_local_model`. |
| `--experiment_name`, `-e` | `10_Clients` | Data experiment directory containing one folder per dataset. |
| `-metrics_list`, `-ml` | empty | Fairness metric to constrain. Repeat the option once per fairness constraint. If omitted, no fairness constraint is created. |
| `-groups_list`, `-gl` | empty | Sensitive group associated with each fairness metric. Must be aligned with `-ml` and `-tl`. |
| `-threshold_list`, `-tl` | empty | Constraint tolerance for each fairness metric/group pair. Must be aligned with `-ml` and `-gl`. |
| `--performance_budget`, `-pb` | omitted | Optional FairLAB performance budget `beta`. If omitted, the performance constraint is not instantiated. |
| `--num_federated_iterations`, `-nf` | `100` | Maximum number of server federated rounds, as in the paper. Training may stop earlier through server-side early stopping. |
| `--random_seed` | `None` | Optional seed for model initialization, client sampling and local training. If omitted, randomness is not seeded. |
| `--validation_strategy` | `holdout` | Persistent holdout or K-fold validation derived only from training data. |
| `--cv_folds` | `5` | Number of folds used by K-fold validation. |
| `--fold_id` | omitted | Fold selected for a K-fold run. |
| `--data_root` | omitted | Common data root containing `10_Clients`; defaults to the repository's `data` directory. |
| `--num_clients`, `-nc` | `10` | Number of federated clients. The builder expects matching split folders `node_1`, ..., `node_n`. |
| `--client_fraction` | `0.5` | Fraction of available clients sampled independently in each federated round. The number selected is rounded up, so 10 clients with `0.5` selects 5. |
| `--clients_per_core`, `-cpc` | `1` | Number of Ray client actors packed on one CPU core. For example, `2` means two clients share one CPU core allocation. |
| `--gpu_devices`, `-g` | omitted | Visible GPU identifiers. Repeat the option for multiple GPUs, e.g. `--gpu_devices 0 --gpu_devices 1`. If omitted, CPU mode is forced. |
| `--checkpoint_dir` | project directory | Optional checkpoint directory override. |
| `--evaluate_test`, `--evaluate-test` | disabled | Evaluate final validation-selected checkpoints on test. |
| `--evaluate_only`, `--evaluate-only` | disabled | Evaluate saved checkpoints without retraining. |
| `--keep_checkpoints` / `--delete_checkpoints` | `--keep_checkpoints` | Keep or remove local checkpoint files after a completed training run. |
| `--debug` | `False` | Enable diagnostic prints and verbose progress bars. By default, diagnostic output is suppressed. |

### Advanced Overrides

The defaults below reproduce the current protocol. Override them only for an
ablation, a different federation layout, or convergence analysis.

| Option | Default | Description |
| --- | --- | --- |
| `--fairlab_orchestrator_epochs`, `--num_global_iterations`, `-ng` | `3` | Number of FairLAB outer/orchestrator epochs on each selected client during the local update phase. |
| `--fairlab_local_epochs`, `--num_local_iterations`, `-nl` | `5` | Number of learner epochs inside each FairLAB orchestrator step. |
| `--performance_step`, `-ps` | `0.10` | FairLAB improvement margin `rho_step` in `P >= p* + rho_step`. |
| `--global_patience`, `-gp` | `10` | Patience for server-side early stopping/checkpoint selection. |
| `--local_patience`, `-lp` | `5` | Patience for client-side local checkpoint selection. |
| `--history_size`, `-hs` | `5` | Maximum number of global models retained in the server teacher history. |
| `--aggregation_epochs`, `-ae` | `10` | Number of server aggregation/distillation rounds after clients return their locally updated models. |
| `--aggregation_local_epochs`, `-ale` | `10` | Number of learner epochs each client runs during aggregation/distillation. |
| `--aggregation-patience` | `2` | Early-stopping patience for validation KD during aggregation. |
| `--aggregation-min-delta` | `1e-6` | Minimum validation-KD improvement. |
| `--validation_fraction` | `0.2` | Fraction reserved for holdout validation. |
| `--stratify_columns` | target + paper-sensitive columns | Columns used for joint stratification. |

## Fairness Constraints

Fairness constraints are passed as aligned repeated arguments:

```bash
-ml <metric_name> -gl <group_name> -tl <threshold>
```

Example matching the paper's three-constraint COMPAS setting:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas_MultiConstraint \
  --id compas_multi \
  --experiment_name 10_Clients \
  --num_clients 4 \
  -ml demographic_parity -gl RaceAge -tl 0.20 \
  -ml demographic_parity -gl Race -tl 0.10 \
  -ml demographic_parity -gl Age -tl 0.10
```

Group names must match the sensitive attributes defined by the selected run.
Paper settings use `Age` and `RaceAge` for COMPAS, and `Marital` and
`RaceMarital` for the other datasets. Individual `Race` constraints are also
used in the multiple-constraint experiments.

## Checkpoints

Checkpoints are written under:

```text
checkpoints/<project_name>/
```

By default these files remain available after training. To upload the final
artifacts to W&B and then remove the local checkpoint directory, run with:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas \
  --id compas_cleanup \
  --num_clients 4 \
  --num_federated_iterations 5 \
  --delete_checkpoints
```

To evaluate existing best global and local checkpoints on the test splits
without retraining, rerun the same experiment configuration with
`--evaluate-only`. The project name, run id, number of clients, dataset run,
fairness metrics/groups/thresholds and data partition must match the training
run:

```bash
python src/main.py \
  --run compas_fedfairlab \
  --project_name FedFairLAB_Compas \
  --id compas_run \
  --num_clients 4 \
  --experiment_name 10_Clients \
  --evaluate-only
```

Use `--checkpoint_dir /path/to/checkpoints` when the saved checkpoints are not
under the default `checkpoints/<project_name>/` directory. Evaluation-only mode
never calls the server or client training loops and never deletes checkpoints.

## Notes for Reproducible Experiments

- Start with a small run (`2` clients, `1-2` server rounds) before launching
  long experiments.
- Keep `--aggregation_epochs 1` and a small `--aggregation_local_epochs` value for quick debugging.
- Add `--debug` only when inspecting resource allocation, client/server flow,
  constraint construction, or local training details.
- Increase `--history_size` only when you want more teacher diversity.
- Use `--clients_per_core` to control Ray CPU packing. For example,
  `--clients_per_core 2` schedules two client actors per CPU core.
- Log outputs and checkpoints depend on the configured `project_name` and `id`.

## Current Implementation Status

The current implementation includes:

- global teacher sampling from server history;
- local FairLAB updates with selective distillation;
- ensemble-logit aggregation from locally updated models;
- federated distillation using `z_ens` with classwise group-robust KL;
- FedAvg over distilled client models;
- FairLAB-style local and global performance budgets;
- support for no-fairness-constraint runs.

The codebase is documented with docstrings throughout `src/` and is intended to
be readable enough for extension and experimentation.

## Citation

If you use FedFairLAB in your research, please cite the following paper:

```bibtex
@article{DBLP:journals/datamine/FontanaNM26,
  author  = {Michele Fontana and Francesca Naretto and Anna Monreale},
  title   = {Federated learning with multiple, intersectional and multiclass fairness guarantees under performance budgets},
  journal = {Data Min. Knowl. Discov.},
  volume  = {40},
  number  = {5},
  pages   = {81},
  year    = {2026},
  doi     = {10.1007/S10618-026-01243-6}
}