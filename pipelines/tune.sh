#!/bin/bash
# Usage: ./pipelines/tune.sh <dataset_name> [options]
# Example: ./pipelines/tune.sh NF-UNSW-NB15-v3 --data_dir ./data
#
# Hyperparameter search using the project's built-in tuning infrastructure.
# Uses main.py --tune with the config_search_space/ tuning YAML files.

set -e

CMD_PREFIX="uv run --python 3.11 python"
DATASET=""
DATA_DIR="./data"
TRIALS=20
TUNE_SPACE="config_search_space/tuning_space.yaml"
GPU=false
RUN_FINAL=false
DRY_RUN=false

usage() {
    echo "Usage: $0 <dataset_name> [options]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name          Name of the dataset (e.g. NF-UNSW-NB15-v3)"
    echo ""
    echo "Options:"
    echo "  --trials N            Number of tuning trials (default: 20)"
    echo "  --tune_space FILE     Tuning space YAML (default: config_search_space/tuning_space.yaml)"
    echo "  --data_dir DIR        Data directory (default: ./data)"
    echo "  --gpu                 Use GPU (auto-detected by default)"
    echo "  --final               Run a final training with best found hyperparameters"
    echo "  --dry-run             Print commands without executing"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 NF-UNSW-NB15-v3 --data_dir ./data --trials 30 --final"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --tune_space)
            TUNE_SPACE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU=true
            shift
            ;;
        --final)
            RUN_FINAL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            if [[ -z "$DATASET" ]]; then
                echo "Error: Unknown option $1"
                usage
            fi
            echo "Error: Unknown option $1"
            usage
            ;;
        *)
            if [[ -z "$DATASET" ]]; then
                DATASET="$1"
                shift
            else
                echo "Error: Unexpected argument $1"
                usage
            fi
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: Dataset name is required."
    usage
fi

if [[ ! -f "$TUNE_SPACE" ]]; then
    echo "Error: Tuning space file not found: $TUNE_SPACE"
    exit 1
fi

if [[ "$DRY_RUN" == false ]]; then
    GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
else
    GPU_AVAILABLE="unknown"
fi

echo "========================================"
echo "  GraphIDS Hyperparameter Tuning"
echo "========================================"
echo "  Dataset:       $DATASET"
echo "  Data dir:      $DATA_DIR"
echo "  Trials:        $TRIALS"
echo "  Tune space:    $TUNE_SPACE"
echo "  GPU available: $GPU_AVAILABLE"
echo "========================================"
echo ""

TUNE_CMD="$CMD_PREFIX main.py --dataset $DATASET --data_dir $DATA_DIR --tune --tune_space $TUNE_SPACE --tune_trials $TRIALS"

echo "--- Hyperparameter search ---"
echo "Command: $TUNE_CMD"
echo ""
if [[ "$DRY_RUN" == false ]]; then
    $TUNE_CMD 2>&1 | tee /tmp/tune_output.txt
    echo ""
    echo "--- Tuning completed ---"
else
    echo "[dry-run] Would execute: $TUNE_CMD"
    echo ""
    echo "[dry-run] (would then parse best params from output)"
fi
echo ""

if [[ "$DRY_RUN" == false ]]; then
    BEST_LINE=$(grep -oP 'best_overrides=\{.*?\}' /tmp/tune_output.txt 2>/dev/null || true)
    BEST_SCORE=$(grep -oP 'best_score=[0-9.+-]+' /tmp/tune_output.txt 2>/dev/null || true)

    echo "========================================"
    echo "  Best Hyperparameters"
    echo "========================================"
    if [[ -n "$BEST_LINE" ]]; then
        echo "  $BEST_LINE"
    else
        echo "  (Could not parse best params from tuning output)"
    fi
    if [[ -n "$BEST_SCORE" ]]; then
        echo "  $BEST_SCORE"
    fi
    echo "========================================"
fi

if [[ "$RUN_FINAL" == true ]]; then
    echo ""
    echo "--- Final training with best hyperparameters ---"
    FINAL_CMD="$CMD_PREFIX main.py --dataset $DATASET --data_dir $DATA_DIR --split_mode temporal --tune --tune_space $TUNE_SPACE --tune_trials $TRIALS"
    echo "Command: $FINAL_CMD"
    if [[ "$DRY_RUN" == false ]]; then
        echo ""
        $FINAL_CMD 2>&1 | tee /tmp/tune_final_output.txt
        echo ""
        echo "--- Final training completed ---"
    else
        echo "[dry-run] Would execute: $FINAL_CMD"
    fi
fi
