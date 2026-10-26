#!/usr/bin/env bash
# ==============================================================================
# Afri-Gloss-MT Master GPU Training Orchestrator
#
# Prevents GPU resource contention:
#   1. Monitors the active multilingual NLLB-600M training job (1.81M samples).
#   2. Once NLLB reaches convergence and finishes, automatically begins
#      the OPUS-MT per-language training queue sequentially across all full datasets.
# ==============================================================================

set -e

PROJECT_DIR="/mnt/volume_d2wey28/projects/afrigloss-mt"
cd "$PROJECT_DIR"

NLLB_PID="${1:-1085425}"
LOG_FILE="$PROJECT_DIR/logs/orchestrator.log"
mkdir -p "$PROJECT_DIR/logs"

echo "$(date) [Orchestrator] Monitoring NLLB PID $NLLB_PID..." | tee -a "$LOG_FILE"

# Wait for NLLB to finish
while kill -0 "$NLLB_PID" 2>/dev/null; do
    sleep 30
done

echo "$(date) [Orchestrator] NLLB-600M training finished!" | tee -a "$LOG_FILE"
echo "$(date) [Orchestrator] Giving GPU 10s cooldown..." | tee -a "$LOG_FILE"
sleep 10

echo "$(date) [Orchestrator] Launching OPUS-MT per-language training queue on full datasets..." | tee -a "$LOG_FILE"
nohup .venv/bin/python scripts/train_100k_queue.py >> "$PROJECT_DIR/logs/train_opus_full_queue.log" 2>&1 &
echo "$(date) [Orchestrator] OPUS queue launched with PID $!" | tee -a "$LOG_FILE"
