#!/usr/bin/env bash
# Combinator E2E Test Runner
#
# Runs combinator-focused suites and a cross-component invariant flow
# (loser-drain + region-quiescence) with deterministic artifact output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RCH_BIN="${RCH_BIN:-rch}"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-300}"

OUTPUT_DIR="${PROJECT_ROOT}/target/e2e-results/combinators"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STARTED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_FILE="${OUTPUT_DIR}/combinators_e2e_${TIMESTAMP}.log"
ARTIFACT_DIR="${OUTPUT_DIR}/artifacts_${TIMESTAMP}"
SUMMARY_MD="${ARTIFACT_DIR}/summary.md"
SUMMARY_JSON="${ARTIFACT_DIR}/summary.json"

export TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-info}"
export RUST_LOG="${RUST_LOG:-${TEST_LOG_LEVEL}}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TEST_SEED="${TEST_SEED:-0xDEADBEEF}"

mkdir -p "$OUTPUT_DIR" "$ARTIFACT_DIR"

if ! command -v "$RCH_BIN" >/dev/null 2>&1; then
    echo "Required executable not found: $RCH_BIN" >&2
    exit 1
fi

echo "==================================================================="
echo "                Asupersync Combinator E2E Tests                   "
echo "==================================================================="
echo ""
echo "Config:"
echo "  TEST_LOG_LEVEL:  ${TEST_LOG_LEVEL}"
echo "  RUST_LOG:        ${RUST_LOG}"
echo "  TEST_SEED:       ${TEST_SEED}"
echo "  Timeout:         ${SUITE_TIMEOUT}s"
echo "  Output:          ${LOG_FILE}"
echo "  Artifacts:       ${ARTIFACT_DIR}"
echo ""

TOTAL_CASES=0
PASSED_CASES=0
FAILED_CASES=0

UNIT_EXIT=0
CANCEL_EXIT=0
ASYNC_EXIT=0
QUIESCENCE_EXIT=0
LOSER_DRAIN_EXIT=0

run_case() {
    local case_label="$1"
    local log_path="$2"
    shift 2

    TOTAL_CASES=$((TOTAL_CASES + 1))
    printf "[%d/5] %-42s" "$TOTAL_CASES" "$case_label"

    set +e
    timeout "$SUITE_TIMEOUT" "$RCH_BIN" exec -- "$@" -- --nocapture --test-threads=1 2>&1 | tee "$log_path"
    local rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
        echo "PASS"
        PASSED_CASES=$((PASSED_CASES + 1))
    else
        echo "FAIL (exit $rc)"
        FAILED_CASES=$((FAILED_CASES + 1))
    fi
    return "$rc"
}

echo ">>> [1/5] Running combinator unit tests..."
run_case \
    "combinator unit tests" \
    "${ARTIFACT_DIR}/unit_tests.log" \
    cargo test --test combinator_tests e2e::combinator::unit || UNIT_EXIT=$?

echo ""
echo ">>> [2/5] Running cancel-correctness tests..."
run_case \
    "cancel correctness" \
    "${ARTIFACT_DIR}/cancel_tests.log" \
    cargo test --test combinator_tests e2e::combinator::cancel_correctness || CANCEL_EXIT=$?

echo ""
echo ">>> [3/5] Running async loser-drain tests..."
run_case \
    "async loser drain" \
    "${ARTIFACT_DIR}/async_tests.log" \
    cargo test --test combinator_tests async_loser_drain || ASYNC_EXIT=$?

echo ""
echo ">>> [4/5] Running cross-component quiescence flow..."
run_case \
    "runtime e2e task spawn + quiescence" \
    "${ARTIFACT_DIR}/runtime_quiescence.log" \
    cargo test --test runtime_e2e e2e_task_spawn_and_quiescence || QUIESCENCE_EXIT=$?

echo ""
echo ">>> [5/5] Running cross-component loser-drain flow..."
run_case \
    "runtime e2e race loser drain" \
    "${ARTIFACT_DIR}/runtime_loser_drain.log" \
    cargo test --test runtime_e2e e2e_race_loser_drain || LOSER_DRAIN_EXIT=$?

cat "${ARTIFACT_DIR}"/*.log > "$LOG_FILE" 2>/dev/null || true

echo ""
echo ">>> [analysis] Checking invariant violation patterns..."
ORACLE_VIOLATIONS=$(grep -hE "(LoserDrainViolation|ObligationLeakViolation|quiescence violation)" "${ARTIFACT_DIR}"/*.log 2>/dev/null | wc -l | tr -d ' ')
if [[ "$ORACLE_VIOLATIONS" -gt 0 ]]; then
    echo "  WARNING: invariant violations detected (${ORACLE_VIOLATIONS})"
    grep -hEn "(LoserDrainViolation|ObligationLeakViolation|quiescence violation)" "${ARTIFACT_DIR}"/*.log | head -20 > "${ARTIFACT_DIR}/invariant_violations.txt" || true
else
    echo "  No invariant violations detected"
fi

PATTERN_FAILURES=$(grep -hE "(panicked at|test result: FAILED|FAILED)" "${ARTIFACT_DIR}"/*.log 2>/dev/null | wc -l | tr -d ' ')

RUN_ENDED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUITE_STATUS="failed"
if [[ "$FAILED_CASES" -eq 0 && "$ORACLE_VIOLATIONS" -eq 0 ]]; then
    SUITE_STATUS="passed"
fi

FAILURE_CLASS="test_or_invariant_failure"
if [[ "$SUITE_STATUS" == "passed" ]]; then
    FAILURE_CLASS="none"
fi

REPRO_COMMAND="TEST_LOG_LEVEL=${TEST_LOG_LEVEL} RUST_LOG=${RUST_LOG} TEST_SEED=${TEST_SEED} SUITE_TIMEOUT=${SUITE_TIMEOUT} bash ${SCRIPT_DIR}/$(basename "$0")"

cat > "$SUMMARY_MD" << EOF
# Combinator E2E Summary

- Timestamp: ${TIMESTAMP}
- Started: ${RUN_STARTED_TS}
- Ended: ${RUN_ENDED_TS}
- Status: ${SUITE_STATUS}

| Case | Exit |
|------|------|
| combinator unit tests | ${UNIT_EXIT} |
| cancel correctness | ${CANCEL_EXIT} |
| async loser drain | ${ASYNC_EXIT} |
| runtime quiescence flow | ${QUIESCENCE_EXIT} |
| runtime loser-drain flow | ${LOSER_DRAIN_EXIT} |

- Total cases: ${TOTAL_CASES}
- Passed cases: ${PASSED_CASES}
- Failed cases: ${FAILED_CASES}
- Invariant violations: ${ORACLE_VIOLATIONS}
- Pattern failures: ${PATTERN_FAILURES}
EOF

cat > "$SUMMARY_JSON" << ENDJSON
{
  "schema_version": "e2e-suite-summary-v3",
  "suite_id": "combinators_e2e",
  "scenario_id": "E2E-SUITE-COMBINATORS",
  "seed": "${TEST_SEED}",
  "started_ts": "${RUN_STARTED_TS}",
  "ended_ts": "${RUN_ENDED_TS}",
  "status": "${SUITE_STATUS}",
  "failure_class": "${FAILURE_CLASS}",
  "repro_command": "${REPRO_COMMAND}",
  "artifact_path": "${SUMMARY_JSON}",
  "suite": "combinators_e2e",
  "timestamp": "${TIMESTAMP}",
  "test_log_level": "${TEST_LOG_LEVEL}",
  "total_cases": ${TOTAL_CASES},
  "passed_cases": ${PASSED_CASES},
  "failed_cases": ${FAILED_CASES},
  "pattern_failures": ${PATTERN_FAILURES},
  "invariant_violations": ${ORACLE_VIOLATIONS},
  "unit_exit": ${UNIT_EXIT},
  "cancel_exit": ${CANCEL_EXIT},
  "async_exit": ${ASYNC_EXIT},
  "quiescence_exit": ${QUIESCENCE_EXIT},
  "loser_drain_exit": ${LOSER_DRAIN_EXIT},
  "log_file": "${LOG_FILE}",
  "artifact_dir": "${ARTIFACT_DIR}",
  "summary_md": "${SUMMARY_MD}",
  "cross_component_flow": {
    "loser_drain": "tests/runtime_e2e.rs::e2e_race_loser_drain",
    "region_quiescence": "tests/runtime_e2e.rs::e2e_task_spawn_and_quiescence"
  }
}
ENDJSON

echo ""
echo "==================================================================="
echo "                  COMBINATOR E2E SUMMARY                          "
echo "==================================================================="
echo "  Cases:    ${PASSED_CASES}/${TOTAL_CASES} passed"
echo "  Invariant violations: ${ORACLE_VIOLATIONS}"
echo "  Pattern failures: ${PATTERN_FAILURES}"
echo "  Summary:  ${SUMMARY_JSON}"
echo "  Artifacts:${ARTIFACT_DIR}"
echo "==================================================================="

if [[ "$SUITE_STATUS" != "passed" ]]; then
    exit 1
fi
