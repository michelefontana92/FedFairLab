import click
import os


def _seed_everything(seed):
    """Seed Python, NumPy and PyTorch when a seed is provided."""
    if seed is None:
        return
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_visible_gpus(gpu_devices):
    """
    Configure CUDA visibility before the run object is constructed.

    Args:
        gpu_devices: Tuple of GPU identifiers provided by the CLI.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        ",".join(gpu_devices) if gpu_devices else "-1"
    )
    if not gpu_devices:
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


def _configure_debug_mode(debug):
    """
    Configure diagnostic printing for the main process and Ray actors.

    Args:
        debug: If true, enable debug prints across the runtime.
    """
    os.environ["FEDFAIRLAB_DEBUG"] = "1" if debug else "0"


def _build_run_kwargs(options):
    """
    Translate CLI option names into the configuration expected by run builders.

    Args:
        options: Dictionary produced by Click.

    Returns:
        Keyword arguments for ``RunFactory.create_run``.
    """
    run_kwargs = {
        "project_name": options["project_name"],
        "id": options["id"],
        "metrics_list": options["metrics_list"],
        "groups_list": options["groups_list"],
        "threshold_list": options["threshold_list"],
        "num_global_iterations": options["fairlab_orchestrator_epochs"],
        "num_local_iterations": options["fairlab_local_epochs"],
        "performance_constraint": options["performance_budget"],
        "performance_step": options["performance_step"],
        "delta": 0.02,
        "max_constraints_in_subproblem": 10000,
        "global_patience": options["global_patience"],
        "local_patience": options["local_patience"],
        "num_clients": options["num_clients"],
        "client_fraction": options["client_fraction"],
        "clients_per_core": options["clients_per_core"],
        "gpu_devices": options["gpu_devices"],
        "verbose": options["debug"],
        "debug": options["debug"],
        "data_root": options["data_root"],
        "num_federated_iterations": options["num_federated_iterations"],
        "history_size": options["history_size"],
        "aggregation_epochs": options["aggregation_epochs"],
        "aggregation_local_epochs": options["aggregation_local_epochs"],
        "aggregation_patience": options["aggregation_patience"],
        "aggregation_min_delta": options["aggregation_min_delta"],
        "experiment_name": options["experiment_name"],
        "keep_checkpoints": options["keep_checkpoints"],
        "evaluate_only": options["evaluate_only"],
        "evaluate_test": options["evaluate_test"],
        "validation_fraction": options["validation_fraction"],
        "random_seed": options["random_seed"],
        "cv_folds": options["cv_folds"],
        "fold_id": options["fold_id"],
    }
    if options["validation_strategy"] is not None:
        run_kwargs["validation_strategy"] = options["validation_strategy"]
    if options["stratify_columns"]:
        run_kwargs["stratify_columns"] = options["stratify_columns"]
    if options["checkpoint_dir"] is not None:
        run_kwargs["checkpoint_dir"] = options["checkpoint_dir"]
    return run_kwargs


@click.command()
# Main experiment options.
@click.option("--run", "-r", default="compas_fedfairlab", help="Run to execute")
@click.option("--project_name", "-p", default="CompasFairLab", help="Project name")
@click.option("--id", "-i", default="test", help="Run id")
@click.option(
    "--experiment_name",
    "-e",
    default="10_Clients",
    show_default=True,
    help="Data experiment directory containing one folder per dataset.",
)
@click.option("-metrics_list", "-ml", multiple=True, help="List of metrics")
@click.option("-groups_list", "-gl", multiple=True, help="List of groups")
@click.option("-threshold_list", "-tl", type=float, multiple=True, help="List of threshold")
@click.option(
    "--performance_budget",
    "-pb",
    type=float,
    default=None,
    help="Optional FairLAB beta budget. If omitted, performance constraints are disabled.",
)
@click.option(
    "--num_federated_iterations",
    "-nf",
    default=100,
    show_default=True,
    help="Number of federated server iterations",
)
@click.option(
    "--random_seed",
    type=int,
    default=None,
    show_default=True,
    help=(
        "Optional seed for model initialization, client sampling, and local "
        "training. By default, randomness is not seeded."
    ),
)
@click.option(
    "--validation_strategy",
    type=click.Choice(["holdout", "kfold"]),
    default=None,
    help="Use a persistent holdout or one K-fold split of the training data.",
)
@click.option(
    "--cv_folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of folds for stratified K-fold validation.",
)
@click.option(
    "--fold_id",
    type=int,
    default=None,
    help="Zero-based validation fold to run when validation_strategy=kfold.",
)
@click.option("--data_root", default=None, help="Optional common dataset root, e.g. /path/to/data.")
@click.option("--num_clients", "-nc", default=10, help="Number of clients")
@click.option(
    "--client_fraction",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    default=0.5,
    show_default=True,
    help="Fraction of available clients sampled in each federated round.",
)
@click.option(
    "--clients_per_core",
    "-cpc",
    default=1,
    help="Number of Ray client actors allowed per CPU core",
)
@click.option("--gpu_devices", "-g", multiple=True, help="List of GPU devices")
@click.option(
    "--checkpoint_dir",
    default=None,
    help="Checkpoint directory. Defaults to checkpoints/<project_name>.",
)
@click.option(
    "--evaluate_test",
    "--evaluate-test",
    is_flag=True,
    default=False,
    help="Evaluate final global and local checkpoints on the test split.",
)
@click.option(
    "--evaluate_only",
    "--evaluate-only",
    is_flag=True,
    default=False,
    help="Evaluate saved global and local checkpoints on test data without training.",
)
@click.option(
    "--keep_checkpoints/--delete_checkpoints",
    default=True,
    help="Keep or remove local checkpoint files after a completed training run.",
)
@click.option("--debug", is_flag=True, default=False, help="Enable diagnostic prints and verbose progress bars.")

# Advanced protocol overrides.
@click.option(
    "--fairlab_orchestrator_epochs",
    "--num_global_iterations",
    "-ng",
    default=3,
    type=int,
    help="Number of FairLAB orchestrator epochs per client update.",
)
@click.option(
    "--fairlab_local_epochs",
    "--num_local_iterations",
    "-nl",
    default=5,
    type=int,
    help="Number of local learner epochs inside each FairLAB orchestrator step.",
)
@click.option(
    "--performance_step",
    "-ps",
    default=0.10,
    show_default=True,
    help="FairLAB rho margin in the improvement constraint P >= p* + rho.",
)
@click.option("--global_patience", "-gp", default=10, help="Global patience")
@click.option("--local_patience", "-lp", default=5, help="Client local patience")
@click.option(
    "--history_size",
    "-hs",
    default=5,
    help="Maximum number of global models kept in server history",
)
@click.option(
    "--aggregation_epochs",
    "-ae",
    type=int,
    default=10,
    help="Number of server aggregation/distillation rounds per federated round.",
)
@click.option(
    "--aggregation_local_epochs",
    "-ale",
    type=int,
    default=10,
    help="Number of local learner epochs run by each client during aggregation distillation.",
)
@click.option(
    "--aggregation-patience",
    type=click.IntRange(min=0),
    default=2,
    show_default=True,
    help=(
        "Stop inner federated distillation after this many iterations without "
        "an improvement in mean client validation KD. Zero disables stopping."
    ),
)
@click.option(
    "--aggregation-min-delta",
    type=click.FloatRange(min=0.0),
    default=1e-6,
    show_default=True,
    help="Minimum mean validation-KD decrease counted as an improvement.",
)
@click.option(
    "--validation_fraction",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction of each client's training pool reserved for holdout validation.",
)
@click.option(
    "--stratify_columns",
    multiple=True,
    help="Raw CSV columns used jointly for holdout/K-fold stratification.",
)
def main(**options):
    """Build and execute a FedFairLab run from CLI options."""
    from runs import RunFactory

    _configure_visible_gpus(options["gpu_devices"])
    _configure_debug_mode(options["debug"])
    _seed_everything(options["random_seed"])
    run = RunFactory.create_run(options["run"], **_build_run_kwargs(options))
    run()


if __name__ == "__main__":
    main()
