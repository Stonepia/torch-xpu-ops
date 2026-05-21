#!/bin/bash
# Batch run R2: 13 issues, each goes start-to-end in one pass
# Phase 1: Smoke test issue #172
# Phase 2: Remaining 12 issues sequentially

set -a
source ~/torch-xpu-ops/.github/issue-handler/.env
set +a

source ~/intel/oneapi/setvars.sh --force 2>/dev/null
source ~/pytorch/.venv/bin/activate

cd ~/torch-xpu-ops/.github/issue-handler

LOGFILE=~/torch-xpu-ops/.github/issue-handler/batch_run_r2.log
SMOKE_ISSUE=172
REMAINING_ISSUES=(171 146 1349 1347 103 270 273 110 137 180 272 111)

echo "========================================" | tee -a "$LOGFILE"
echo "BATCH RUN R2 START: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOGFILE"
echo "Total issues: 13 (1 smoke + 12 batch)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Smoke test
echo "" | tee -a "$LOGFILE"
echo "=== SMOKE TEST: #$SMOKE_ISSUE ===" | tee -a "$LOGFILE"
echo "[$(date -u '+%H:%M:%S')] Processing #$SMOKE_ISSUE (full pipeline)..." | tee -a "$LOGFILE"
python scripts/run_pipeline.py --once --issues "$SMOKE_ISSUE" --stages format triage fix verify_fix 2>&1 | tee -a "$LOGFILE"
SMOKE_EXIT=$?
echo "[$(date -u '+%H:%M:%S')] #$SMOKE_ISSUE done (exit=$SMOKE_EXIT)" | tee -a "$LOGFILE"

if [ $SMOKE_EXIT -ne 0 ]; then
    echo "❌ SMOKE TEST FAILED — aborting batch" | tee -a "$LOGFILE"
    exit 1
fi
echo "✅ Smoke test passed — proceeding with batch" | tee -a "$LOGFILE"

# Batch
echo "" | tee -a "$LOGFILE"
echo "=== BATCH: ${#REMAINING_ISSUES[@]} issues ===" | tee -a "$LOGFILE"
for issue in "${REMAINING_ISSUES[@]}"; do
    echo "[$(date -u '+%H:%M:%S')] Processing #$issue (full pipeline)..." | tee -a "$LOGFILE"
    python scripts/run_pipeline.py --once --issues "$issue" --stages format triage fix verify_fix 2>&1 | tee -a "$LOGFILE"
    STATUS=$?
    echo "[$(date -u '+%H:%M:%S')] #$issue done (exit=$STATUS)" | tee -a "$LOGFILE"
    echo "---" | tee -a "$LOGFILE"
done

# Dashboard update
echo "" | tee -a "$LOGFILE"
echo "[$(date -u '+%H:%M:%S')] Updating dashboard..." | tee -a "$LOGFILE"
ALL_ISSUES=($SMOKE_ISSUE ${REMAINING_ISSUES[@]})
python scripts/run_pipeline.py --once --batch "Run 2 — 13 issues" --issues "${ALL_ISSUES[@]}" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "BATCH RUN R2 COMPLETE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
