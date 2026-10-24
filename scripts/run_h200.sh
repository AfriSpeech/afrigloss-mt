#!/usr/bin/env bash
# ==============================================================================
# Afri-Gloss-MT H200 Remote Execution Helper
# Syncs code to the Ghana NLP H200 GPU instance and launches jobs.
#
# Usage:
#   ./scripts/run_h200.sh sync                     # Sync code to H200
#   ./scripts/run_h200.sh align <lang>             # Run full alignment on H200
#   ./scripts/run_h200.sh align-all                # Run alignment on priority African languages
#   ./scripts/run_h200.sh train <src> <tgt> <data> # Run fine-tuning on H200
# ==============================================================================

set -e

REMOTE_HOST="h200"
REMOTE_DIR="/mnt/volume_d2wey28/projects/afrigloss-mt"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

function sync_to_h200() {
    echo "==> Syncing afrigloss-mt to H200 ($REMOTE_DIR)..."
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='checkpoints' \
          --exclude='.pytest_cache' --exclude='*.pyc' \
          "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
    echo "==> Sync complete."
}

ACTION="${1:-sync}"

case "$ACTION" in
    sync)
        sync_to_h200
        ;;
    align)
        LANG_CODE="${2:-twi}"
        sync_to_h200
        echo "==> Running word alignment for $LANG_CODE on H200..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && python3 scripts/prerun_alignments.py --langs $LANG_CODE"
        echo "==> Fetching extracted dictionary back to local..."
        rsync -avz "$REMOTE_HOST:$REMOTE_DIR/data/dictionaries/" "$LOCAL_DIR/data/dictionaries/"
        ;;
    align-all)
        sync_to_h200
        echo "==> Running word alignment for all priority African languages on H200..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && python3 scripts/prerun_alignments.py"
        echo "==> Fetching extracted dictionaries back to local..."
        rsync -avz "$REMOTE_HOST:$REMOTE_DIR/data/dictionaries/" "$LOCAL_DIR/data/dictionaries/"
        ;;
    train)
        SRC="${2:-en}"
        TGT="${3:-tw}"
        DATA_DIR="${4:-data/gloss_dataset}"
        sync_to_h200
        echo "==> Launching fine-tuning for $SRC -> $TGT on H200 Hopper GPU (bf16+TF32)..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && python3 -m afrigloss.cli train --src $SRC --tgt $tgt --train-file $DATA_DIR/train.jsonl --val-file $DATA_DIR/validation.jsonl --out checkpoints/afrigloss_${SRC}_${TGT}"
        ;;
    *)
        echo "Unknown command: $ACTION"
        echo "Available: sync, align <lang>, align-all, train <src> <tgt> <data>"
        exit 1
        ;;
esac
