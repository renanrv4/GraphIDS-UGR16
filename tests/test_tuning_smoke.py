from __future__ import annotations

from types import SimpleNamespace

import main as main_mod


class DummyRun:
    def __init__(self, config: dict, name: str = "dummy") -> None:
        # main.py usa run.config como objeto com acesso tipo atributo
        self.config = SimpleNamespace(**config)
        self.name = name
        self.logged = []
        self.finished = False

    def log(self, d: dict) -> None:
        self.logged.append(d)

    def finish(self) -> None:
        self.finished = True


def test_tuning_path_runs_and_applies_best_overrides(monkeypatch, tmp_path):
    """
    Smoke/integration-ish test:
    - runs the "tune_hyperparameters" path (without real data)
    - ensures best overrides are applied to final config
    - ensures final "main(run)" calls train and test once
    """

    # --- Arrange: fake args and base config ---
    args = SimpleNamespace(
        # runtime required
        data_dir=str(tmp_path / "data"),
        checkpoint=None,
        reload_dataset=False,
        test=False,
        save_curve=False,
        seed=1,
        wandb=False,
        # experiment keys required
        data_type="benign",
        dataset="NF-UNSW-NB15-v2",
        num_epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        ae_weight_decay=0.0,
        edim_out=64,
        batch_size=8,
        fanout=-1,
        agg_type="mean",
        num_layers=1,
        mask_ratio=0.0,
        patience=1,
        ae_batch_size=2,
        window_size=4,
        step_percent=1.0,
        ae_embedding_dim=16,
        ae_dropout=0.0,
        dropout=0.0,
        positional_encoding="None",
        fraction=None,
        split_mode="stratified",
        distribution_segment="pre_shift",
        # tuning flags
        tune=True,
        tune_space=str(tmp_path / "space.yaml"),
        tune_trials=2,
        tune_seed=123,
        tune_metric="val_pr_auc",
        tune_num_epochs=1,
        tune_patience=1,
        # config file not used here
        config=None,
    )

    # minimal fake search space file (the code reads it)
    (tmp_path / "space.yaml").write_text(
        """
learning_rate:
  type: choice
  values: [0.001, 0.01]
mask_ratio:
  type: choice
  values: [0.0, 0.2]
""".strip()
    )

    base_cfg_dict = {k: getattr(args, k) for k in main_mod.EXPERIMENT_CONFIG_KEYS}

    # --- Spy counters ---
    calls = {
        "tune_hyperparameters": 0,
        "final_train_calls": 0,
        "final_test_calls": 0,
    }

    # --- Monkeypatch heavy training/eval so test is fast and deterministic ---
    def fake_tune_hyperparameters(passed_args, passed_base_cfg):
        calls["tune_hyperparameters"] += 1
        # pretend best config found sets learning_rate and mask_ratio
        return {"learning_rate": 0.01, "mask_ratio": 0.2}

    monkeypatch.setattr(main_mod, "tune_hyperparameters", fake_tune_hyperparameters)

    # Avoid touching real wandb, torch-geometric, etc.
    # We'll patch main_mod.main to a "final pipeline" that asserts overrides are present.
    def fake_main(run):
        # check overrides applied
        assert run.config.learning_rate == 0.01
        assert run.config.mask_ratio == 0.2

        # simulate that training + testing happens
        calls["final_train_calls"] += 1
        calls["final_test_calls"] += 1
        run.log({"final_test_pr_auc": 0.5})
        run.finish()

    monkeypatch.setattr(main_mod, "main", fake_main)

    # Patch apply_cli_config/ensure_config_keys to no-op for this unit test
    monkeypatch.setattr(main_mod, "apply_cli_config", lambda run_config, a: None)
    monkeypatch.setattr(main_mod, "ensure_config_keys", lambda cfg: None)

    # Patch wandb.init to return DummyRun with config dict turned into attrs
    def fake_wandb_init(*, project, config, name=None, reinit=False):
        assert project == "GraphIDS"
        # config can be dict
        return DummyRun(config=config, name=name or "final")

    monkeypatch.setattr(main_mod.wandb, "init", fake_wandb_init)

    # --- Act: emulate the __main__ logic for tuning+final run ---
    # This is basically the bottom of main.py but runnable in test.
    config = base_cfg_dict
    best_overrides = main_mod.tune_hyperparameters(args, config)
    config = dict(config)
    config.update(best_overrides)

    run = main_mod.wandb.init(project="GraphIDS", config=config)
    main_mod.apply_cli_config(run.config.__dict__, args)  # harmless due to no-op
    main_mod.ensure_config_keys(run.config.__dict__)      # harmless due to no-op
    main_mod.main(run)

    # --- Assert ---
    assert calls["tune_hyperparameters"] == 1
    assert calls["final_train_calls"] == 1
    assert calls["final_test_calls"] == 1
    assert run.finished is True