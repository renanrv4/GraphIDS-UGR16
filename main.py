import os
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader

import patch_pyg
patch_pyg.apply()

from models.graphids import GraphIDS
from utils.dataloaders import NetFlowDataset, StreamingNetFlowDataset
from utils.parser import Parser
from utils.trainers import (
    test,
    train,
    train_online,
    train_temporal_windows,
    update_threshold_online,
    validate,
)

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


# =========================
# NEW TUNING HELPERS    
# =========================
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


def _run_train_val_once(
    run, *, dataset, train_loader, val_loader, checkpoint_suffix: str | None = None
):
    """
    Runs train+val (and will still construct test_loader, but we can avoid calling test()).
    Returns: (best_val_pr_auc, trained_model, threshold, checkpoint_path)
    """
    config = run.config

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
            {"params": model.encoder.parameters(), "weight_decay": config.weight_decay},
            {
                "params": model.transformer.parameters(),
                "weight_decay": config.ae_weight_decay,
            },
        ],
        lr=config.learning_rate,
    )

    # checkpoint per trial to avoid overwriting
    base_ckpt = resolve_checkpoint_path(config, run.name)
    if checkpoint_suffix:
        root, ext = os.path.splitext(base_ckpt)
        checkpoint = f"{root}_{checkpoint_suffix}{ext or '.ckpt'}"
    else:
        checkpoint = base_ckpt

    checkpoint_dir = os.path.dirname(checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

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

    return best_val_pr_auc, model, threshold, checkpoint


def tune_hyperparameters(args, dataset, base_config_dict: dict) -> dict:
    if not args.tune_space:
        raise ValueError("--tune requires --tune_space <path.yaml>")
    base_config_dict = dict(base_config_dict)
    if not isinstance(base_config_dict, dict):
        raise ValueError("base_config_dict must be a dict")

    with open(args.tune_space, encoding="utf-8") as f:
        space = yaml.safe_load(f)
    if not isinstance(space, dict):
        raise ValueError("tune_space YAML must be a mapping of hyperparameter -> spec")

    rng = random.Random(args.tune_seed)

    best_overrides = None
    best_score = -float("inf")

    shuffle = base_config_dict["positional_encoding"] == "None"
    fanout_list = (
        [base_config_dict["fanout"]] if base_config_dict["fanout"] != -1 else [-1]
    )

    train_loader = LinkNeighborLoader(
        data=dataset.train_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.train_graph.edge_index,
        edge_label=dataset.train_graph.edge_labels,
        batch_size=base_config_dict["batch_size"],
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = LinkNeighborLoader(
        data=dataset.val_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.val_graph.edge_index,
        edge_label=dataset.val_graph.edge_labels,
        batch_size=base_config_dict["batch_size"],
        shuffle=shuffle,
        drop_last=True,
    )

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
            trial_run,
            dataset=dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_suffix=f"trial{trial}",
        )

        print(f"[tuning] trial={trial} val_pr_auc={score:.6f} params={overrides}")

        if not np.isnan(score) and score > best_score:
            best_score = score
            best_overrides = overrides

        trial_run.finish()

        print(
            f"[tuning] trial {trial}/{args.tune_trials} overrides={overrides} score={score}"
        )

    print(f"[tuning] best_overrides={best_overrides} best_score={best_score}")
    return best_overrides, train_loader, val_loader



# =========================
# TRAINER HELPER
# =========================

def train_model(
    run,
    dataset,
    tune,
    resume_train,
    train_loader=None,
    val_loader=None,
):
    config = run.config

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
            {
                "params": model.encoder.parameters(),
                "weight_decay": config.weight_decay,
            },
            {
                "params": model.transformer.parameters(),
                "weight_decay": config.ae_weight_decay,
            },
        ],
        lr=config.learning_rate,
    )

    checkpoint = resolve_checkpoint_path(config, run.name)

    checkpoint_dir = os.path.dirname(checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ==================================================
    # Resume training only when called
    # ==================================================
    if resume_train and os.path.exists(checkpoint):
        print("Loading model from checkpoint")

        start_epoch, threshold = model.load_checkpoint(
            checkpoint,
            optimizer,
        )

        run.config.epoch = start_epoch

    else:
        start_epoch = 0
        threshold = None

    # ==================================================
    # Load config
    # ==================================================
    shuffle = config.positional_encoding == "None"
    fanout_list = [config.fanout] if config.fanout != -1 else [-1]

    recommended_workers = 0

    if not tune:
        train_loader = LinkNeighborLoader(
            data=dataset.train_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.train_graph.edge_index,
            edge_label=dataset.train_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

        val_loader = LinkNeighborLoader(
            data=dataset.val_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.val_graph.edge_index,
            edge_label=dataset.val_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    print("Starting training...")

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

    return (
        model,
        threshold,
        start_epoch,
        fanout_list,
        shuffle,
        recommended_workers,
        best_val_pr_auc,
    )


def run_streaming(args, run=None):
    if run is None:
        config = build_wandb_config(args)
        if not args.wandb:
            os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(project="GraphIDS", config=config)
        apply_cli_config(run.config, args)
        ensure_config_keys(run.config)
    config = run.config

    set_seed(config.seed)

    dataset = StreamingNetFlowDataset(
        name=config.dataset,
        data_dir=config.data_dir,
        force_reload=config.reload_dataset,
        fraction=config.fraction,
        data_type=config.data_type,
        seed=config.seed,
    )

    edim_in = dataset.windows[0].edge_attr.size(1)
    ndim_in = dataset.windows[0].x.size(1)

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
            {"params": model.encoder.parameters(), "weight_decay": config.weight_decay},
            {
                "params": model.transformer.parameters(),
                "weight_decay": config.ae_weight_decay,
            },
        ],
        lr=config.learning_rate,
    )

    fanout_list = [config.fanout] if config.fanout != -1 else [-1]
    shuffle = config.positional_encoding == "None"
    num_windows = dataset.num_windows
    split_idx = int(num_windows * 0.6)

    print(f"Total temporal windows: {num_windows}")
    print(f"Initial training on first {split_idx} windows")

    for i in range(split_idx):
        window_data = dataset.get_window(i)
        loader = LinkNeighborLoader(
            data=window_data,
            num_neighbors=fanout_list,
            edge_label_index=window_data.edge_index,
            edge_label=window_data.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        model, _, _ = train(
            model,
            config.window_size,
            config.step_percent,
            config.ae_batch_size,
            loader,
            loader,
            0,
            max(5, config.num_epochs // num_windows),
            optimizer,
            run,
            config.patience,
            None,
            device=device,
        )

    replay_buffer = [dataset.get_window(i) for i in range(split_idx)]
    threshold = None

    print(f"Streaming evaluation on remaining {num_windows - split_idx} windows")
    for i in range(split_idx, num_windows):
        window_data = dataset.get_window(i)
        val_loader = LinkNeighborLoader(
            data=window_data,
            num_neighbors=fanout_list,
            edge_label_index=window_data.edge_index,
            edge_label=window_data.edge_labels,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        _, errors, labels = validate(
            model, val_loader, config.ae_batch_size, config.window_size, device
        )

        threshold = update_threshold_online(
            errors,
            labels,
            method="validation_f1",
            prev_threshold=threshold,
            alpha=0.1,
        )

        preds = (errors > threshold).int()
        f1 = f1_score(labels.cpu(), preds.cpu(), average="macro", zero_division=0)
        pr_auc = average_precision_score(labels.cpu(), errors.cpu())

        run.log({
            f"stream_window_{i}_f1": f1,
            f"stream_window_{i}_pr_auc": pr_auc,
            f"stream_window_{i}_threshold": threshold,
        })

    run.finish()
    return model


def main(run, args):
    if args.streaming:
        return run_streaming(args, run=run)

    config = run.config
    print(config)

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

    if args.tune:
        best_overrides, train_loader, val_loader = tune_hyperparameters(
            args, dataset, config
        )
        config = build_wandb_config(args)

        if isinstance(config, str):
            with open(config, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            config = {
                k: v["value"] if isinstance(v, dict) and "value" in v else v
                for k, v in config.items()
            }

        config.update(best_overrides)

        run = wandb.init(project="GraphIDS", config=config)

        apply_cli_config(run.config, args)
        ensure_config_keys(run.config)
        config = run.config
        (
            model,
            threshold,
            start_epoch,
            fanout_list,
            shuffle,
            recommended_workers,
            _,
        ) = train_model(run, dataset, True, False, train_loader, val_loader)
    else:
        (
            model,
            threshold,
            start_epoch,
            fanout_list,
            shuffle,
            recommended_workers,
            _,
        ) = train_model(run, dataset, False, True, None, None)

    if start_epoch >= config.num_epochs or config.test:
        print("Model already trained")
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
    else:
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge.edge_labels
            if hasattr(dataset.test_graph, "edge")
            else dataset.test_graph.edge_labels,  # keep compatibility
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=False,
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
    print(f"FINAL_TEST_F1: {test_f1:.6f}")
    print(f"FINAL_TEST_PR_AUC: {test_pr_auc:.6f}")
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

    run = wandb.init(project="GraphIDS", config=config)
    apply_cli_config(run.config, args)
    ensure_config_keys(run.config)

    main(run, args)
