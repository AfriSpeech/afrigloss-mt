#!/bin/bash
# Launch Master Pan-African Dataset, Dictionary, and Proper Noun Pipeline

cd /mnt/volume_d2wey28/projects/afrigloss-mt || exit 1
mkdir -p logs

# Wait for any prior test run to complete cleanly
while pgrep -f "build_pan_african_master.py --only" > /dev/null; do
    echo "[$(date)] Waiting for single-language test run to finish..."
    sleep 10
done

echo "[$(date)] Starting full Pan-African Master Pipeline in background..."
nohup nice -n 10 .venv/bin/python scripts/build_pan_african_master.py > logs/pan_african_master_nohup.log 2>&1 &
MAIN_PID=$!
echo "[$(date)] Pan-African Master Pipeline launched with PID: $MAIN_PID"
echo $MAIN_PID > logs/pan_african_master.pid
