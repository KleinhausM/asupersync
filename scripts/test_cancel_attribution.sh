#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RCH_BIN="${RCH_BIN:-rch}"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-240}"

OUTPUT_DIR="${PROJECT_ROOT}/target/e2e-results/cancel-attribution"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STARTED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ARTIFACT_DIR="${OUTPUT_DIR}/artifacts_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/cancel_attribution_e2e_${TIMESTAMP}.log"
SUMMARY_TXT="${ARTIFACT_DIR}/summary.txt"
SUMMARY_JSON="${ARTIFACT_DIR}/summary.json"

mkdir -p "$OUTPUT_DIR" "$ARTIFACT_DIR"
: > "$SUMMARY_TXT"

export TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-trace}"
export RUST_LOG="${RUST_LOG:-trace}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TEST_SEED="${TEST_SEED:-0xDEADBEEF}"

if ! command -v "$RCH_BIN" >/dev/null 2>&1; then
    echo "Required executable not found: $RCH_BIN" >&2
    exit 1
fi

echo "==================================================================="
echo "               Cancel Attribution E2E Test Suite                  "
echo "==================================================================="
echo "  TEST_LOG_LEVEL: ${TEST_LOG_LEVEL}"
echo "  RUST_LOG:       ${RUST_LOG}"
echo "  TEST_SEED:      ${TEST_SEED}"
echo "  Timeout:        ${SUITE_TIMEOUT}s"
echo "  Artifacts:      ${ARTIFACT_DIR}"
echo ""

TOTAL_CASES=0
PASSED_CASES=0
FAILED_CASES=0

run_test() {
    local name="$1"
    local pattern="$2"
    local log_path="${ARTIFACT_DIR}/${name}.log"
    TOTAL_CASES=$((TOTAL_CASES + 1))

    printf "[%d] %-38s" "$TOTAL_CASES" "$name"

    set +e
    timeout "$SUITE_TIMEOUT" "$RCH_BIN" exec -- cargo test --test cancel_attribution "$pattern" -- --nocapture --test-threads=1 2>&1 | tee "$log_path"
    local rc=$?
    set -e

    local passed_count
    local failed_count
    passed_count=$(grep -c "^test .* ok$" "$log_path" 2>/dev/null || true)
    failed_count=$(grep -c "^test .* FAILED$" "$log_path" 2>/dev/null || true)

    if [[ "$rc" -eq 0 ]]; then
        echo "PASS"
        PASSED_CASES=$((PASSED_CASES + 1))
        echo "PASS  ${name} (${passed_count} tests)" >> "$SUMMARY_TXT"
    else
        echo "FAIL (exit $rc)"
        FAILED_CASES=$((FAILED_CASES + 1))
        echo "FAIL  ${name} (${failed_count} failures, exit ${rc})" >> "$SUMMARY_TXT"
    fi
}

echo ">>> CancelReason construction"
run_test "cancel_reason_construction" "cancel_reason_basic_construction"
run_test "cancel_reason_builder" "cancel_reason_builder_methods"

echo ""
echo ">>> Cause-chain behavior"
run_test "cause_chain_construction" "cancel_reason_cause_chain_construction"
run_test "root_cause" "cancel_reason_root_cause"
run_test "any_cause_is" "cancel_reason_any_cause_is"

echo ""
echo ">>> CancelKind behavior"
run_test "cancel_kind_variants" "cancel_kind_all_variants_constructible"
run_test "cancel_kind_eq_hash" "cancel_kind_eq_and_hash"

echo ""
echo ">>> Cx API behavior"
run_test "cx_cancel_with" "cx_cancel_with_stores_reason"
run_test "cx_cancel_with_no_msg" "cx_cancel_with_no_message"
run_test "cx_cancel_chain" "cx_cancel_chain_api"
run_test "cx_root_cancel_cause" "cx_root_cancel_cause_api"
run_test "cx_cancelled_by" "cx_cancelled_by_api"
run_test "cx_any_cause_is" "cx_any_cause_is_api"
run_test "cx_cancel_fast" "cx_cancel_fast_api"

echo ""
echo ">>> E2E behavior"
run_test "e2e_debugging_workflow" "e2e_debugging_workflow"
run_test "e2e_metrics_collection" "e2e_metrics_collection"
run_test "e2e_severity_handling" "e2e_severity_based_handling"
run_test "integration_handler_usage" "integration_realistic_handler_usage"

cat "${ARTIFACT_DIR}"/*.log > "$LOG_FILE" 2>/dev/null || true

PATTERN_FAILURES=$(grep -hE "(panicked at|test result: FAILED|FAILED)" "${ARTIFACT_DIR}"/*.log 2>/dev/null | wc -l | tr -d ' ')
RUN_ENDED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

SUITE_STATUS="failed"
if [[ "$FAILED_CASES" -eq 0 && "$PATTERN_FAILURES" -eq 0 ]]; then
    SUITE_STATUS="passed"
fi

FAILURE_CLASS="test_or_pattern_failure"
if [[ "$SUITE_STATUS" == "passed" ]]; then
    FAILURE_CLASS="none"
fi

REPRO_COMMAND="TEST_LOG_LEVEL=${TEST_LOG_LEVEL} RUST_LOG=${RUST_LOG} TEST_SEED=${TEST_SEED} SUITE_TIMEOUT=${SUITE_TIMEOUT} bash ${SCRIPT_DIR}/$(basename "$0")"

cat > "$SUMMARY_JSON" << ENDJSON
{
  "schema_version": "e2e-suite-summary-v3",
  "suite_id": "cancel_attribution_e2e",
  "scenario_id": "E2E-SUITE-CANCEL-ATTRIBUTION",
  "seed": "${TEST_SEED}",
  "started_ts": "${RUN_STARTED_TS}",
  "ended_ts": "${RUN_ENDED_TS}",
  "status": "${SUITE_STATUS}",
  "failure_class": "${FAILURE_CLASS}",
  "repro_command": "${REPRO_COMMAND}",
  "artifact_path": "${SUMMARY_JSON}",
  "suite": "cancel_attribution_e2e",
  "timestamp": "${TIMESTAMP}",
  "test_log_level": "${TEST_LOG_LEVEL}",
  "total_cases": ${TOTAL_CASES},
  "passed_cases": ${PASSED_CASES},
  "failed_cases": ${FAILED_CASES},
  "pattern_failures": ${PATTERN_FAILURES},
  "summary_txt": "${SUMMARY_TXT}",
  "log_file": "${LOG_FILE}",
  "artifact_dir": "${ARTIFACT_DIR}"
}
ENDJSON

echo ""
echo "==================================================================="
echo "                         TEST SUMMARY                             "
echo "==================================================================="
cat "$SUMMARY_TXT"
echo "-------------------------------------------------------------------"
echo "Cases passed:  ${PASSED_CASES}"
echo "Cases failed:  ${FAILED_CASES}"
echo "Pattern failures: ${PATTERN_FAILURES}"
echo "Summary: ${SUMMARY_JSON}"
echo "Artifacts: ${ARTIFACT_DIR}"
echo "==================================================================="

if [[ "$SUITE_STATUS" != "passed" ]]; then
    exit 1
fi
