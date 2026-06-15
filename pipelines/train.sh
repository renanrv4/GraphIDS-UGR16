#!/bin/bash
# Usage: ./pipelines/train.sh <dataset_name> [options]
# Example: ./pipelines/train.sh NF-UNSW-NB15-v3 --data_dir ./data
#
# Runs:
#   1. Batch mode training (baseline)
#   2. Streaming mode training
#   3. Comparison of results

set -e

CMD_PREFIX="uv run --python 3.11 python"
DATASET=""
DATA_DIR="./data"
NUM_EPOCHS=50
GPU=false
DRY_RUN=false

usage() {
    echo "Usage: $0 <dataset_name> [options]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name          Name of the dataset (e.g. NF-UNSW-NB15-v3)"
    echo ""
    echo "Options:"
    echo "  --data_dir DIR        Data directory (default: ./data)"
    echo "  --num_epochs N        Number of training epochs (default: 50)"
    echo "  --gpu                 Use GPU (auto-detected by default)"
    echo "  --dry-run             Print commands without executing"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 NF-UNSW-NB15-v3 --data_dir ./data --num_epochs 100 --gpu"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --gpu)
            GPU=true
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

if [[ "$DRY_RUN" == false ]]; then
    GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
else
    GPU_AVAILABLE="unknown"
fi

echo "========================================"
echo "  GraphIDS Training Pipeline"
echo "========================================"
echo "  Dataset:       $DATASET"
echo "  Data dir:      $DATA_DIR"
echo "  Num epochs:    $NUM_EPOCHS"
echo "  GPU available: $GPU_AVAILABLE"
echo "========================================"
echo ""

BATCH_CMD="$CMD_PREFIX main.py --dataset $DATASET --data_dir $DATA_DIR --split_mode temporal --num_epochs $NUM_EPOCHS"
STREAM_CMD="$CMD_PREFIX main.py --dataset $DATASET --data_dir $DATA_DIR --streaming --num_epochs $NUM_EPOCHS"

echo "--- Step 1: Batch mode training ---"
echo "Command: $BATCH_CMD"
echo ""
if [[ "$DRY_RUN" == false ]]; then
    $BATCH_CMD 2>&1 | tee /tmp/train_batch_output.txt
    echo ""
    echo "--- Batch mode completed ---"
else
    echo "[dry-run] Would execute: $BATCH_CMD"
fi
echo ""

echo "--- Step 2: Streaming mode training ---"
echo "Command: $STREAM_CMD"
echo ""
if [[ "$DRY_RUN" == false ]]; then
    $STREAM_CMD 2>&1 | tee /tmp/train_stream_output.txt
    echo ""
    echo "--- Streaming mode completed ---"
else
    echo "[dry-run] Would execute: $STREAM_CMD"
fi
echo ""

echo "========================================"
echo "  Results Comparison"
echo "========================================"
if [[ "$DRY_RUN" == false ]]; then
    BATCH_F1=$(grep -oP 'Test macro F1-score: \K[0-9.]+' /tmp/train_batch_output.txt 2>/dev/null || echo "N/A")
    BATCH_PRAUC=$(grep -oP 'Test PR-AUC: \K[0-9.]+' /tmp/train_batch_output.txt 2>/dev/null || echo "N/A")
    BATCH_TIME=$(grep -oP 'Test prediction time: \K[0-9.]+' /tmp/train_batch_output.txt 2>/dev/null || echo "N/A")
    STREAM_F1=$(grep -oP 'stream_window_\d+_f1: \K[0-9.]+' /tmp/train_stream_output.txt 2>/dev/null | tail -1 || echo "N/A")
    STREAM_PRAUC=$(grep -oP 'stream_window_\d+_pr_auc: \K[0-9.]+' /tmp/train_stream_output.txt 2>/dev/null | tail -1 || echo "N/A")

    printf "%-25s %-15s %-15s\n" "Mode" "F1-Score" "PR-AUC"
    printf "%-25s %-15s %-15s\n" "------------------------" "---------------" "---------------"
    printf "%-25s %-15s %-15s\n" "Batch (temporal split)" "$BATCH_F1" "$BATCH_PRAUC"
    printf "%-25s %-15s %-15s\n" "Streaming (temporal)" "$STREAM_F1" "$STREAM_PRAUC"
    echo ""
    echo "Batch test prediction time: ${BATCH_TIME}s"
else
    echo "[dry-run] No results to compare."
fi
echo "========================================"
