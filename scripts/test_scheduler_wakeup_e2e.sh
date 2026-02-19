#!/usr/bin/env bash
# E2E Test Script for Scheduler Wakeup & Race Condition Verification
#
# Runs the full scheduler test pyramid:
#   1. Unit tests (parker, wake state, queues, stealing)
#   2. Lane fairness tests
#   3. Stress tests (high contention, work stealing, backoff)
#   4. Loom systematic concurrency tests (if loom cfg available)
#   5. Cross-component invariant flows (region quiescence + loser drain)
#
# Usage:
#   ./scripts/test_scheduler_wakeup_e2e.sh
#
# Environment Variables:
#   SKIP_STRESS    - Set to 1 to skip stress tests
#   SKIP_LOOM      - Set to 1 to skip Loom tests
#   STRESS_TIMEOUT - Timeout for stress tests in seconds (default: 180)
#   RUST_LOG       - Standard Rust logging level

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RCH_BIN="${RCH_BIN:-rch}"

OUTPUT_DIR="${PROJECT_ROOT}/target/e2e-results/scheduler"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STARTED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_DIR="${OUTPUT_DIR}/artifacts_${TIMESTAMP}"
STRESS_TIMEOUT="${STRESS_TIMEOUT:-180}"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-240}"
LOG_FILE="${OUTPUT_DIR}/scheduler_e2e_${TIMESTAMP}.log"
SUMMARY_MD="${LOG_DIR}/summary.md"
SUMMARY_JSON="${LOG_DIR}/summary.json"

export RUST_LOG="${RUST_LOG:-info}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-info}"
export TEST_SEED="${TEST_SEED:-0xDEADBEEF}"

mkdir -p "$LOG_DIR"

TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

if ! command -v "$RCH_BIN" >/dev/null 2>&1; then
    echo "Required executable not found: $RCH_BIN" >&2
    exit 1
fi

echo "==================================================================="
echo "       Scheduler Wakeup E2E Test Suite                             "
echo "==================================================================="
echo ""
echo "Configuration:"
echo "  Log directory:   $LOG_DIR"
echo "  Stress timeout:  ${STRESS_TIMEOUT}s"
echo "  Suite timeout:   ${SUITE_TIMEOUT}s"
echo "  Seed:            ${TEST_SEED}"
echo "  Skip stress:     ${SKIP_STRESS:-no}"
echo "  Skip loom:       ${SKIP_LOOM:-no}"
echo "  Start time:      $(date -Iseconds)"
echo ""

run_suite() {
    local name="$1"
    local log_file="$LOG_DIR/${name}.log"
    shift
    TOTAL_SUITES=$((TOTAL_SUITES + 1))

    echo "[$TOTAL_SUITES] Running $name..."
    set +e
    timeout "$SUITE_TIMEOUT" "$@" 2>&1 | tee "$log_file"
    local rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
        echo "    PASS"
        PASSED_SUITES=$((PASSED_SUITES + 1))
        return 0
    else
        echo "    FAIL (exit $rc, see $log_file)"
        FAILED_SUITES=$((FAILED_SUITES + 1))
        return "$rc"
    fi
}

# --------------------------------------------------------------------------
# 1. Scheduler unit tests (parker, queues, stealing, backoff)
# --------------------------------------------------------------------------
run_suite "scheduler_backoff" \
    "$RCH_BIN" exec -- cargo test --test scheduler_backoff -- --nocapture --test-threads=1 || true

# --------------------------------------------------------------------------
# 2. Lane fairness tests
# --------------------------------------------------------------------------
run_suite "scheduler_lane_fairness" \
    "$RCH_BIN" exec -- cargo test --test scheduler_lane_fairness -- --nocapture --test-threads=1 || true

# --------------------------------------------------------------------------
# 3. Stress tests (ignored by default, need --ignored flag)
# --------------------------------------------------------------------------
if [ "${SKIP_STRESS:-0}" != "1" ]; then
    run_suite "stress_tests" \
        timeout "${STRESS_TIMEOUT}s" \
        "$RCH_BIN" exec -- cargo test --release scheduler_stress -- --ignored --nocapture --test-threads=1 || true
else
    echo "[skip] Stress tests (SKIP_STRESS=1)"
fi

# --------------------------------------------------------------------------
# 4. Loom systematic concurrency tests
# --------------------------------------------------------------------------
if [ "${SKIP_LOOM:-0}" != "1" ]; then
    run_suite "loom_tests" \
        "$RCH_BIN" exec -- cargo test --test scheduler_loom --features loom-tests --release -- --nocapture --test-threads=1 || true
else
    echo "[skip] Loom tests (SKIP_LOOM=1)"
fi

# --------------------------------------------------------------------------
# 5. Cross-component invariant flows
# --------------------------------------------------------------------------
run_suite "runtime_quiescence_flow" \
    "$RCH_BIN" exec -- cargo test --test runtime_e2e e2e_task_spawn_and_quiescence -- --nocapture --test-threads=1 || true

run_suite "runtime_loser_drain_flow" \
    "$RCH_BIN" exec -- cargo test --test runtime_e2e e2e_race_loser_drain -- --nocapture --test-threads=1 || true

# --------------------------------------------------------------------------
# Failure pattern analysis
# --------------------------------------------------------------------------
echo ""
echo ">>> Analyzing logs for issues..."
ISSUES=0

for pattern in "timed out" "timeout" "deadlock" "hung" "blocked forever"; do
    if grep -rqi "$pattern" "$LOG_DIR"/*.log 2>/dev/null; then
        echo "  WARNING: '$pattern' detected"
        ISSUES=$((ISSUES + 1))
    fi
done

if grep -rq "lost wakeup" "$LOG_DIR"/*.log 2>/dev/null; then
    echo "  WARNING: Lost wakeup detected"
    ISSUES=$((ISSUES + 1))
fi

if grep -rq "double schedule\|duplicate" "$LOG_DIR"/*.log 2>/dev/null; then
    echo "  WARNING: Double scheduling detected"
    ISSUES=$((ISSUES + 1))
fi

if grep -rq "panicked at" "$LOG_DIR"/*.log 2>/dev/null; then
    echo "  WARNING: Panics detected"
    grep -rh "panicked at" "$LOG_DIR"/*.log | head -5
    ISSUES=$((ISSUES + 1))
fi

cat "$LOG_DIR"/*.log > "$LOG_FILE" 2>/dev/null || true

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
RUN_ENDED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUITE_STATUS="failed"
if [[ "$FAILED_SUITES" -eq 0 && "$ISSUES" -eq 0 ]]; then
    SUITE_STATUS="passed"
fi
FAILURE_CLASS="suite_or_pattern_failure"
if [[ "$SUITE_STATUS" == "passed" ]]; then
    FAILURE_CLASS="none"
fi
REPRO_COMMAND="TEST_LOG_LEVEL=${TEST_LOG_LEVEL} RUST_LOG=${RUST_LOG} TEST_SEED=${TEST_SEED} STRESS_TIMEOUT=${STRESS_TIMEOUT} SUITE_TIMEOUT=${SUITE_TIMEOUT} bash ${SCRIPT_DIR}/$(basename "$0")"

cat > "$SUMMARY_MD" << EOF
# Scheduler Wakeup E2E Test Report

**Started:** ${RUN_STARTED_TS}
**Ended:** ${RUN_ENDED_TS}

## Results

| Suite | Status |
|-------|--------|
| Total | $TOTAL_SUITES |
| Passed | $PASSED_SUITES |
| Failed | $FAILED_SUITES |
| Issues | $ISSUES |

## Test Counts
$(grep -rh "^test result:" "$LOG_DIR"/*.log 2>/dev/null || echo "N/A")

## Failures
$(grep -rhE "(FAILED|panicked)" "$LOG_DIR"/*.log 2>/dev/null | head -20 || echo "None")
EOF

cat > "$SUMMARY_JSON" << ENDJSON
{
  "schema_version": "e2e-suite-summary-v3",
  "suite_id": "scheduler_e2e",
  "scenario_id": "E2E-SUITE-SCHEDULER-WAKEUP",
  "seed": "${TEST_SEED}",
  "started_ts": "${RUN_STARTED_TS}",
  "ended_ts": "${RUN_ENDED_TS}",
  "status": "${SUITE_STATUS}",
  "failure_class": "${FAILURE_CLASS}",
  "repro_command": "${REPRO_COMMAND}",
  "artifact_path": "${SUMMARY_JSON}",
  "suite": "scheduler_e2e",
  "timestamp": "${TIMESTAMP}",
  "test_log_level": "${TEST_LOG_LEVEL}",
  "total_suites": ${TOTAL_SUITES},
  "passed_suites": ${PASSED_SUITES},
  "failed_suites": ${FAILED_SUITES},
  "issues": ${ISSUES},
  "log_file": "${LOG_FILE}",
  "artifact_dir": "${LOG_DIR}",
  "summary_md": "${SUMMARY_MD}",
  "cross_component_flow": {
    "loser_drain": "tests/runtime_e2e.rs::e2e_race_loser_drain",
    "region_quiescence": "tests/runtime_e2e.rs::e2e_task_spawn_and_quiescence"
  }
}
ENDJSON

echo ""
echo "==================================================================="
echo "                       SUMMARY                                     "
echo "==================================================================="
echo "  Suites:  $PASSED_SUITES/$TOTAL_SUITES passed"
echo "  Issues:  $ISSUES pattern warnings"
echo "  Logs:    $LOG_DIR/"
echo "  Summary: $SUMMARY_JSON"
echo "  End:     $(date -Iseconds)"
echo "==================================================================="

if [ "$SUITE_STATUS" != "passed" ]; then
    exit 1
fi

echo ""
echo "All scheduler wakeup tests passed!"
