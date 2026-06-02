import os
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader

from models.graphids import GraphIDS
from utils.dataloaders import NetFlowDataset
from utils.parser import Parser
from utils.trainers import test, train

# Suppress this warning: even if in prototype stage, it works correctly for our use case
warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT_CONFIG_KEYS = (
    "data_type",
    "dataset",
    "num_epochs",
    "learning_rate",
    "weight_decay",
    "ae_weight_decay",
    "edim_out",
    "batch_size",
    "fanout",
    "agg_type",
    "num_layers",
    "mask_ratio",
    "patience",
    "ae_batch_size",
    "window_size",
    "step_percent",
    "ae_embedding_dim",
    "ae_dropout",
    "dropout",
    "positional_encoding",
    "fraction",
    "split_mode",
    "distribution_segment",
)
RUNTIME_CONFIG_KEYS = (
    "data_dir",
    "checkpoint",
    "reload_dataset",
    "test",
    "save_curve",
    "seed",
    "wandb",
    # tuning keys live in args only (not required in run.config)
)
BACKWARD_COMPATIBLE_DEFAULT_KEYS = (
    "split_mode",
    "distribution_segment",
)
REQUIRED_CONFIG_KEYS = EXPERIMENT_CONFIG_KEYS + RUNTIME_CONFIG_KEYS


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_wandb_config(args):
    if args.config is not None:
        return args.config
    return {key: getattr(args, key) for key in EXPERIMENT_CONFIG_KEYS}


def apply_cli_config(run_config, args):
    for key in RUNTIME_CONFIG_KEYS:
        run_config[key] = getattr(args, key)

    for key in BACKWARD_COMPATIBLE_DEFAULT_KEYS:
        if key not in run_config:
            run_config[key] = getattr(args, key)


def ensure_config_keys(config):
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"Missing required configuration values: {missing}")


def default_checkpoint_path(config, run_name):
    checkpoint_id = run_name if config.wandb else config.seed
    checkpoint_dataset = config.dataset
    if config.split_mode != "stratified":
        checkpoint_dataset = f"{checkpoint_dataset}_{config.split_mode}"
        if config.split_mode == "temporal_shift_aware":
            checkpoint_dataset = f"{checkpoint_dataset}_{config.distribution_segment}"
    return f"checkpoints/GraphIDS_{checkpoint_dataset}_{checkpoint_id}.ckpt"


def resolve_checkpoint_path(config, run_name):
    if config.checkpoint is not None:
        return config.checkpoint
    return default_checkpoint_path(config, run_name)


# ------------------------ NEW: tuning helpers ------------------------
def _sample_from_space(space: dict, rng: random.Random) -> dict:
    """
    space YAML format example:
      learning_rate:
        type: loguniform
        min: 1e-4
        max: 3e-3
      ae_embedding_dim:
        type: choice
        values: [16, 32, 64]
      num_layers:
        type: int
        min: 1
        max: 3
    """
    sampled = {}
    for key, spec in space.items():
        stype = spec.get("type")
        if stype == "choice":
            vals = spec["values"]
            sampled[key] = rng.choice(vals)
        elif stype == "int":
            sampled[key] = rng.randint(int(spec["min"]), int(spec["max"]))
        elif stype == "uniform":
            sampled[key] = rng.uniform(float(spec["min"]), float(spec["max"]))
        elif stype == "loguniform":
            lo = float(spec["min"])
            hi = float(spec["max"])
            # sample in log10-space
            x = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            sampled[key] = float(x)
        else:
            raise ValueError(f"Unknown search space type for '{key}': {stype}")
    return sampled


def _apply_trial_overrides(base_config, overrides: dict):
    """
    wandb config is an object; we override keys in-place.
    """
    for k, v in overrides.items():
        if k not in EXPERIMENT_CONFIG_KEYS:
            raise KeyError(
                f"Search space key '{k}' is not an experiment hyperparameter "
                f"(allowed: {', '.join(EXPERIMENT_CONFIG_KEYS)})"
            )
        base_config[k] = v


def _run_train_val_once(run, *, checkpoint_suffix: str | None = None):
    """
    Runs train+val (and will still construct test_loader, but we can avoid calling test()).
    Returns: (best_val_pr_auc, trained_model, threshold, checkpoint_path)
    """
    config = run.config
    set_seed(config.seed)

    # dataset = NetFlowDataset(
    #     name=config.dataset,
    #     data_dir=config.data_dir,
    #     force_reload=config.reload_dataset,
    #     fraction=config.fraction,
    #     data_type=config.data_type,
    #     seed=config.seed,
    #     split_mode=config.split_mode,
    #     distribution_segment=config.distribution_segment,
    # )

    # ndim_in = dataset.num_node_features
    # edim_in = dataset.num_edge_features
    # print("Number of features:", edim_in)

    # model = GraphIDS(
    #     ndim_in=ndim_in,
    #     edim_in=edim_in,
    #     edim_out=config.edim_out,
    #     embed_dim=config.ae_embedding_dim,
    #     num_heads=4,
    #     num_layers=config.num_layers,
    #     window_size=config.window_size,
    #     dropout=config.dropout,
    #     ae_dropout=config.ae_dropout,
    #     positional_encoding=config.positional_encoding,
    #     agg_type=config.agg_type,
    #     mask_ratio=config.mask_ratio,
    # ).to(device)

    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": model.encoder.parameters(), "weight_decay": config.weight_decay},
    #         {
    #             "params": model.transformer.parameters(),
    #             "weight_decay": config.ae_weight_decay,
    #         },
    #     ],
    #     lr=config.learning_rate,
    # )

    # # checkpoint per trial to avoid overwriting
    # base_ckpt = resolve_checkpoint_path(config, run.name)
    # if checkpoint_suffix:
    #     root, ext = os.path.splitext(base_ckpt)
    #     checkpoint = f"{root}_{checkpoint_suffix}{ext or '.ckpt'}"
    # else:
    #     checkpoint = base_ckpt

    # checkpoint_dir = os.path.dirname(checkpoint)
    # if checkpoint_dir:
    #     os.makedirs(checkpoint_dir, exist_ok=True)

    # shuffle = config.positional_encoding == "None"
    # fanout_list = [config.fanout] if config.fanout != -1 else [-1]
    # cpu_count = os.cpu_count()
    # recommended_workers = min(cpu_count, 6) if cpu_count is not None else 0

    # train_loader = LinkNeighborLoader(
    #     data=dataset.train_graph,
    #     num_neighbors=fanout_list,
    #     edge_label_index=dataset.train_graph.edge_index,
    #     edge_label=dataset.train_graph.edge_labels,
    #     batch_size=config.batch_size,
    #     shuffle=shuffle,
    #     num_workers=recommended_workers,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     drop_last=True,
    # )
    # val_loader = LinkNeighborLoader(
    #     data=dataset.val_graph,
    #     num_neighbors=fanout_list,
    #     edge_label_index=dataset.val_graph.edge_index,
    #     edge_label=dataset.val_graph.edge_labels,
    #     batch_size=config.batch_size,
    #     shuffle=shuffle,
    #     num_workers=recommended_workers,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     drop_last=True,
    # )

    print("Starting training (train+val)...")
    start_epoch = 0
    model, threshold, best_val_pr_auc = train(
        model,
        config.window_size,
        config.step_percent,
        config.ae_batch_size,
        train_loader,
        val_loader,
        start_epoch,
        config.num_epochs,
        optimizer,
        run,
        config.patience,
        checkpoint,
        device=device,
    )

    # train() logs val_pr_auc during training, but we also want a scalar to compare trials.
    # Simplest: read the last logged val_pr_auc from run history isn't available here.
    # So we rely on the fact that the checkpoint saved is tied to best val PR-AUC,
    # and train() loads best checkpoint at end (see utils/trainers.py) and returns it.
    #
    # We'll compute PR-AUC on validation one final time by reusing the validate routine indirectly:
    # easiest without importing validate: run test() against val_loader-like semantics isn't supported.
    #
    # Minimal approach: use run.summary if W&B online; offline this isn't.
    # So we choose a deterministic proxy: set best_val_pr_auc from the last metric logged into run,
    # by having train() log it; W&B stores it in run._history (private) not safe.
    #
    # Practical approach: during tuning, we use W&B and inspect "val_pr_auc" from the log files.
    #
    # To keep this code self-contained, we return NaN and still store best config in run.config.
    return best_val_pr_auc, model, threshold, checkpoint


def tune_hyperparameters(args, base_config_dict: dict) -> dict:
    if not args.tune_space:
        raise ValueError("--tune requires --tune_space <path.yaml>")

    with open(args.tune_space, "r", encoding="utf-8") as f:
        space = yaml.safe_load(f)
    if not isinstance(space, dict):
        raise ValueError("tune_space YAML must be a mapping of hyperparameter -> spec")

    rng = random.Random(args.tune_seed)

    best_overrides = None
    best_score = -float("inf")

    # Use offline mode unless user explicitly requested online logging
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    for trial in range(1, args.tune_trials + 1):
        overrides = _sample_from_space(space, rng)

        # Build a per-trial config dict and optionally override epochs/patience for tuning only
        trial_cfg = deepcopy(base_config_dict)
        trial_cfg.update(overrides)
        if args.tune_num_epochs is not None:
            trial_cfg["num_epochs"] = args.tune_num_epochs
        if args.tune_patience is not None:
            trial_cfg["patience"] = args.tune_patience

        trial_run = wandb.init(
            project="GraphIDS",
            config=trial_cfg,
            name=f"tune_trial_{trial}",
            reinit=True,
        )
        # Apply runtime keys (data_dir, seed, etc.)
        apply_cli_config(trial_run.config, args)
        ensure_config_keys(trial_run.config)

        # Run training (train+val). NOTE: best_val_pr_auc is NaN in this minimal implementation.
        score, _, _, _ = _run_train_val_once(
            trial_run, checkpoint_suffix=f"trial{trial}"
        )

        print(
            f"[tuning] trial={trial} "
            f"val_pr_auc={score:.6f} "
            f"params={overrides}"
        )

        # If you want a real numeric comparison without depending on W&B internals,
        # you can extend utils/trainers.py to return best_val_pr_auc directly and use it here.
        # For now, we will compare using "score" only if it's a number.
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_overrides = overrides

        trial_run.finish()

        print(f"[tuning] trial {trial}/{args.tune_trials} overrides={overrides} score={score}")

    # Fallback: if we couldn't compute scores programmatically, just return the last sampled overrides
    if best_overrides is None:
        print(
            "[tuning] Could not compute val_pr_auc score in-code; "
            "falling back to the last sampled configuration. "
            "For proper selection, extend trainers.py to return best val_pr_auc."
        )
        best_overrides = overrides  # last trial

    print(f"[tuning] best_overrides={best_overrides} best_score={best_score}")
    return best_overrides
# -------------------------------------------------------------------


def main(run, tune):
    config = run.config

    set_seed(config.seed)
    split_mode = config.split_mode
    distribution_segment = config.distribution_segment

    dataset = NetFlowDataset(
        name=config.dataset,
        data_dir=config.data_dir,
        force_reload=config.reload_dataset,
        fraction=config.fraction,
        data_type=config.data_type,
        seed=config.seed,
        split_mode=split_mode,
        distribution_segment=distribution_segment,
    )

    ndim_in = dataset.num_node_features
    edim_in = dataset.num_edge_features
    print("Number of features:", edim_in)

    model = GraphIDS(
        ndim_in=ndim_in,
        edim_in=edim_in,
        edim_out=config.edim_out,
        embed_dim=config.ae_embedding_dim,
        num_heads=4,
        num_layers=config.num_layers,
        window_size=config.window_size,
        dropout=config.dropout,
        ae_dropout=config.ae_dropout,
        positional_encoding=config.positional_encoding,
        agg_type=config.agg_type,
        mask_ratio=config.mask_ratio,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {  # Higher weight decay for embedding layer
                "params": model.encoder.parameters(),
                "weight_decay": config.weight_decay,
            },
            {  # Lower weight decay for the transformer
                "params": model.transformer.parameters(),
                "weight_decay": config.ae_weight_decay,
            },
        ],
        lr=config.learning_rate,
    )
    checkpoint = resolve_checkpoint_path(config, run.name)
    if os.path.exists(checkpoint):
        print("Loading model from checkpoint")
        start_epoch, threshold = model.load_checkpoint(checkpoint, optimizer)
        run.config.epoch = start_epoch
    else:
        checkpoint_dir = os.path.dirname(checkpoint)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        start_epoch = 0
        threshold = None

    shuffle = config.positional_encoding == "None"
    fanout_list = [config.fanout] if config.fanout != -1 else [-1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cpu_count = os.cpu_count()
    recommended_workers = min(cpu_count, 6) if cpu_count is not None else 0
    if start_epoch >= config.num_epochs or config.test:
        print("Model already trained")
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
    else:
        train_loader = LinkNeighborLoader(
            data=dataset.train_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.train_graph.edge_index,
            edge_label=dataset.train_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader = LinkNeighborLoader(
            data=dataset.val_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.val_graph.edge_index,
            edge_label=dataset.val_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge.edge_labels if hasattr(dataset.test_graph, "edge") else dataset.test_graph.edge_labels,  # keep compatibility
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
        if tune:
            base_cfg_dict = config if isinstance(config, dict) else {key: getattr(args, key) for key in EXPERIMENT_CONFIG_KEYS}
            best_overrides = tune_hyperparameters(args, base_cfg_dict)

            # Apply best overrides to the final run's config
            if isinstance(config, dict):
                config.update(best_overrides)
            else:
                # shouldn't happen after the YAML load above, but kept for safety
                config = {key: getattr(args, key) for key in EXPERIMENT_CONFIG_KEYS}
                config.update(best_overrides)
        print("Starting training...")
        model, threshold = train(
            model,
            config.window_size,
            config.step_percent,
            config.ae_batch_size,
            train_loader,
            val_loader,
            start_epoch,
            config.num_epochs,
            optimizer,
            run,
            config.patience,
            checkpoint,
            device=device,
        )

    print("Evaluating on test set...")
    test_f1, test_pr_auc, errors, test_labels, prediction_time = test(
        model,
        test_loader,
        config.ae_batch_size,
        config.window_size,
        device,
        threshold=threshold,
    )
    precision, recall, _ = precision_recall_curve(test_labels.cpu(), errors.cpu())
    if config.save_curve:
        run.log(
            {
                "Precision-Recall Curve": wandb.plot.pr_curve(
                    y_true=test_labels.cpu().numpy(),
                    y_probas=errors.cpu().numpy(),
                    title="Precision-Recall Curve",
                ),
            }
        )
        os.makedirs("curves", exist_ok=True)
        np.savez(
            f"curves/precision_recall_{run.name}.npz",
            precision=precision,
            recall=recall,
        )
    test_pred = (errors > threshold).int()
    print(f"Test macro F1-score: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    print(f"Test prediction time: {prediction_time:.4f} seconds")
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB")
    else:
        peak_memory_mb = 0
    run.log(
        {
            "final_test_f1": test_f1,
            "final_test_pr_auc": test_pr_auc,
            "test_threshold": threshold,
            "test_prediction_time": prediction_time,
            "peak_gpu_memory_mb": peak_memory_mb,
            "Test Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=test_labels.ravel().tolist(),
                preds=test_pred.ravel().tolist(),
                class_names=["Benign", "Malicious"],
                title="Test Confusion Matrix",
            ),
        }
    )
    run.finish()


if __name__ == "__main__":
    args = Parser().parse_args()

    # Base config comes from YAML (--config) or CLI defaults
    config = build_wandb_config(args)
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # # If config is a YAML path, wandb.init will load it.
    # # For tuning we need a dict, so when args.config is provided, we keep tuning disabled unless you parse YAML yourself.
    # # Minimal: if args.tune and args.config is a file, we load it now into a dict.
    # if args.tune and isinstance(config, str):
    #     if args.tune and isinstance(config, str):
    #         with open(config, "r", encoding="utf-8") as f:
    #             config = yaml.safe_load(f)

    #         config = {
    #             key: value["value"]
    #             if isinstance(value, dict) and "value" in value
    #             else value
    #             for key, value in config.items()
    #         }

    # # ------------------- NEW: optional tuning before final run -------------------
    # if args.tune:
    #     base_cfg_dict = config if isinstance(config, dict) else {key: getattr(args, key) for key in EXPERIMENT_CONFIG_KEYS}
    #     best_overrides = tune_hyperparameters(args, base_cfg_dict)

    #     # Apply best overrides to the final run's config
    #     if isinstance(config, dict):
    #         config.update(best_overrides)
    #     else:
    #         # shouldn't happen after the YAML load above, but kept for safety
    #         config = {key: getattr(args, key) for key in EXPERIMENT_CONFIG_KEYS}
    #         config.update(best_overrides)
    # # ---------------------------------------------------------------------------

    run = wandb.init(project="GraphIDS", config=config)
    apply_cli_config(run.config, args)
    ensure_config_keys(run.config)

    main(run)