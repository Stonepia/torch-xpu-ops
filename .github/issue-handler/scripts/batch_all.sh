#!/bin/bash
# Batch-run all 49 issues through the full pipeline (format → triage → fix → verify_fix)
# Runs in tmux, logs to batch_run.log, updates dashboard #1694.
# Monitor: tail -f ~/torch-xpu-ops/.github/issue-handler/batch_run.log

set -a
source ~/torch-xpu-ops/.github/issue-handler/.env
set +a

# Ensure oneAPI environment
source ~/intel/oneapi/setvars.sh 2>/dev/null

cd ~/torch-xpu-ops/.github/issue-handler

LOGFILE=~/torch-xpu-ops/.github/issue-handler/batch_run.log
ISSUES=(346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 1658 1659 1660 1661 1662 1663 1666 1668 1693 1695)

echo "========================================" | tee -a "$LOGFILE"
echo "BATCH RUN START: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOGFILE"
echo "Total issues: ${#ISSUES[@]}" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Phase 1: format + triage all
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 1: FORMAT + TRIAGE ===" | tee -a "$LOGFILE"
for issue in "${ISSUES[@]}"; do
    echo "[$(date -u '+%H:%M:%S')] Processing #$issue (format+triage)..." | tee -a "$LOGFILE"
    python scripts/run_pipeline.py --once --issues "$issue" --stages format triage 2>&1 | tee -a "$LOGFILE"
    echo "[$(date -u '+%H:%M:%S')] #$issue format+triage done" | tee -a "$LOGFILE"
    echo "---" | tee -a "$LOGFILE"
done

# Update dashboard after phase 1
echo "" | tee -a "$LOGFILE"
echo "[$(date -u '+%H:%M:%S')] Updating dashboard after Phase 1..." | tee -a "$LOGFILE"
python scripts/run_pipeline.py --once --batch 2>&1 | tee -a "$LOGFILE"

# Phase 2: fix (code_fix)
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 2: CODE FIX ===" | tee -a "$LOGFILE"
for issue in "${ISSUES[@]}"; do
    echo "[$(date -u '+%H:%M:%S')] Processing #$issue (fix)..." | tee -a "$LOGFILE"
    python scripts/run_pipeline.py --once --issues "$issue" --stages fix 2>&1 | tee -a "$LOGFILE"
    echo "[$(date -u '+%H:%M:%S')] #$issue fix done" | tee -a "$LOGFILE"
    echo "---" | tee -a "$LOGFILE"
done

# Update dashboard after phase 2
echo "" | tee -a "$LOGFILE"
echo "[$(date -u '+%H:%M:%S')] Updating dashboard after Phase 2..." | tee -a "$LOGFILE"
python scripts/run_pipeline.py --once --batch 2>&1 | tee -a "$LOGFILE"

# Phase 3: verify_fix
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 3: VERIFY FIX ===" | tee -a "$LOGFILE"
for issue in "${ISSUES[@]}"; do
    echo "[$(date -u '+%H:%M:%S')] Processing #$issue (verify_fix)..." | tee -a "$LOGFILE"
    python scripts/run_pipeline.py --once --issues "$issue" --stages verify_fix 2>&1 | tee -a "$LOGFILE"
    echo "[$(date -u '+%H:%M:%S')] #$issue verify_fix done" | tee -a "$LOGFILE"
    echo "---" | tee -a "$LOGFILE"
done

# Final dashboard update
echo "" | tee -a "$LOGFILE"
echo "=== FINAL DASHBOARD UPDATE ===" | tee -a "$LOGFILE"
python scripts/run_pipeline.py --once --batch 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "BATCH RUN COMPLETE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
