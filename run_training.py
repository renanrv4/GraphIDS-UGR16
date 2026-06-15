#!/usr/bin/env python3
"""
Runner de treino GraphIDS com status file e métricas periódicas.

Uso:
  uv run --python 3.11 --frozen python run_training.py [batch|streaming|both]

Progresso:
  cat training_status.json   — estado atual do treino
  tail -f training_runner.log — logs completos em tempo real

Modos:
  batch     — treino batch com split temporal (padrão)
  streaming — treino streaming com janelas temporais
  both      — batch + streaming sequencial
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "training_status.json"
CMD_PREFIX = ["uv", "run", "--python", "3.11", "--frozen", "python", "main.py"]


def load_status():
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {"started_at": None, "finished_at": None, "steps": {}}


def save_status(status):
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))


def run_step(step_name, extra_args, status):
    step = {
        "status": "running",
        "started_at": str(datetime.now()),
        "pid": None,
        "epochs": [],
        "last_val_pr_auc": None,
        "best_val_pr_auc": None,
    }
    status["steps"][step_name] = step
    save_status(status)

    cmd = CMD_PREFIX + extra_args
    print(f"\n{'='*70}")
    print(f"  Iniciando: {step_name}")
    print(f"  Comando: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    step["pid"] = proc.pid
    save_status(status)

    for line in proc.stdout:
        print(line, end="", flush=True)

        # Época: "EPOCH: N/30 | train_loss=X | val_loss=Y | val_pr_auc=Z | best_val_pr_auc=W"
        m = re.search(
            r"EPOCH:\s*(\d+)/(\d+)\s*\|.*val_pr_auc=([\d\.eE+-]+)\s*\|.*best_val_pr_auc=([\d\.eE+-]+)",
            line,
        )
        if m:
            epoch_info = {
                "epoch": int(m.group(1)),
                "total": int(m.group(2)),
                "val_pr_auc": float(m.group(3)),
                "best_val_pr_auc": float(m.group(4)),
            }
            step["epochs"].append(epoch_info)
            step["last_val_pr_auc"] = epoch_info["val_pr_auc"]
            step["best_val_pr_auc"] = epoch_info["best_val_pr_auc"]
            save_status(status)

        # Métricas finais batch
        m = re.search(r"FINAL_TEST_F1:\s*([\d\.]+)", line)
        if m:
            step["test_f1"] = float(m.group(1))
            save_status(status)

        m = re.search(r"FINAL_TEST_PR_AUC:\s*([\d\.]+)", line)
        if m:
            step["test_pr_auc"] = float(m.group(1))
            save_status(status)

        # Métricas streaming
        m = re.search(r"stream_window_(\d+)_f1:\s*([\d\.]+)", line)
        if m:
            step["last_window_f1"] = float(m.group(2))
            save_status(status)

        m = re.search(r"stream_window_(\d+)_pr_auc:\s*([\d\.]+)", line)
        if m:
            step["last_window_pr_auc"] = float(m.group(2))
            save_status(status)

    ret = proc.wait()
    step["status"] = "completed" if ret == 0 else "failed"
    step["exit_code"] = ret
    step["finished_at"] = str(datetime.now())
    save_status(status)

    if ret != 0:
        print(f"\n[ERRO] Step '{step_name}' falhou com código {ret}")
    return ret


def show_periodic_summary(status):
    for name, step in status.get("steps", {}).items():
        if step.get("status") == "running":
            ep = step.get("epochs", [])
            if ep:
                last = ep[-1]
                print(
                    f"  [{name}] epoch {last['epoch']}/{last['total']} | "
                    f"val_pr_auc={last['val_pr_auc']:.4f} | "
                    f"best={last['best_val_pr_auc']:.4f}"
                )
            else:
                print(f"  [{name}] processando dados...")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"

    status = load_status()
    status["started_at"] = str(datetime.now())
    status["dataset"] = "NF-UNSW-NB15-v3"
    status["fraction"] = 0.3
    status["mode"] = mode
    save_status(status)

    BASE_ARGS = [
        "--dataset", "NF-UNSW-NB15-v3",
        "--data_dir", "./data",
        "--fraction", "0.3",
        "--num_epochs", "30",
        "--batch_size", "8192",
        "--ae_batch_size", "32",
        "--learning_rate", "0.0001",
        "--weight_decay", "0.6",
        "--ae_weight_decay", "0.04",
        "--edim_out", "96",
        "--ae_embedding_dim", "48",
        "--num_layers", "1",
        "--mask_ratio", "0.15",
        "--window_size", "512",
        "--dropout", "0.6",
        "--ae_dropout", "0.0",
        "--patience", "20",
        "--seed", "42",
        "--step_percent", "1.0",
    ]

    steps = []
    if mode in ("batch", "both"):
        steps.append(("batch_temporal", BASE_ARGS + ["--split_mode", "temporal"]))
    if mode in ("streaming", "both"):
        steps.append(("streaming", BASE_ARGS + ["--streaming"]))

    all_ok = True
    for step_name, args in steps:
        ret = run_step(step_name, args, status)
        if ret != 0:
            all_ok = False
            if mode != "both":
                break

    status["finished_at"] = str(datetime.now())
    status["all_ok"] = all_ok
    save_status(status)

    print(f"\n{'='*70}")
    print(f"  Treino finalizado em {status['finished_at']}")
    if all_ok:
        print(f"  Status: OK")
        for name, step in status.get("steps", {}).items():
            tf1 = step.get("test_f1")
            tpr = step.get("test_pr_auc")
            if tf1 is not None:
                print(f"  [{name}] F1={tf1:.4f}  PR-AUC={tpr:.4f}")
    else:
        print(f"  Status: COM FALHAS")
    print(f"  Status file: cat {STATUS_FILE}")
    print(f"  Logs: tail -f training_runner.log")
    print(f"{'='*70}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
