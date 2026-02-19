#!/usr/bin/env bash
# Network Hardening E2E Test Runner (C3)
#
# Runs the full network hardening pyramid with deterministic settings and
# structured forensic artifacts.
#
# Usage:
#   ./scripts/test_net_hardening_e2e.sh
#
# Environment Variables:
#   TEST_LOG_LEVEL - error|warn|info|debug|trace (default: trace)
#   RUST_LOG       - tracing filter (default: asupersync=debug)
#   RUST_BACKTRACE - 1 to enable backtraces (default: 1)
#   TEST_SEED      - deterministic seed override (default: 0xDEADBEEF)
#   SUITE_TIMEOUT  - per-suite timeout in seconds (default: 180)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/target/e2e-results/net_hardening"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STARTED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ARTIFACT_DIR="${OUTPUT_DIR}/artifacts_${TIMESTAMP}"
LOG_DIR="${ARTIFACT_DIR}/logs"
SUMMARY_JSON="${ARTIFACT_DIR}/summary.json"
SUMMARY_MD="${ARTIFACT_DIR}/summary.md"
MANIFEST_FILE="${ARTIFACT_DIR}/artifact_manifest.json"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-180}"

export TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-trace}"
export RUST_LOG="${RUST_LOG:-asupersync=debug}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TEST_SEED="${TEST_SEED:-0xDEADBEEF}"

mkdir -p "$OUTPUT_DIR" "$ARTIFACT_DIR" "$LOG_DIR"

TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

echo "==================================================================="
echo "       Network Primitives Hardening E2E Test Suite                 "
echo "==================================================================="
echo ""
echo "Config:"
echo "  TEST_LOG_LEVEL:  ${TEST_LOG_LEVEL}"
echo "  RUST_LOG:        ${RUST_LOG}"
echo "  TEST_SEED:       ${TEST_SEED}"
echo "  Timeout:         ${SUITE_TIMEOUT}s"
echo "  Timestamp:       ${TIMESTAMP}"
echo "  Artifacts:       ${ARTIFACT_DIR}"
echo "  Logs:            ${LOG_DIR}"
echo ""

echo ">>> [1/5] Pre-flight: checking compilation..."
if ! cargo check --tests --all-features 2>"${ARTIFACT_DIR}/compile_errors.log"; then
    echo "  FATAL: compilation failed - see ${ARTIFACT_DIR}/compile_errors.log"
    exit 1
fi
echo "  OK"

run_suite() {
    local name="$1"
    local log_file="$LOG_DIR/${name}.log"
    shift
    TOTAL_SUITES=$((TOTAL_SUITES + 1))

    echo "[$TOTAL_SUITES] Running $name..."
    if timeout "$SUITE_TIMEOUT" "$@" 2>&1 | tee "$log_file"; then
        echo "    PASS"
        PASSED_SUITES=$((PASSED_SUITES + 1))
        return 0
    else
        echo "    FAIL (see $log_file)"
        FAILED_SUITES=$((FAILED_SUITES + 1))
        return 1
    fi
}

echo ""
echo ">>> [2/5] Running network hardening suites..."

run_suite "tcp_unit" cargo test --lib net::tcp -- --nocapture || true
run_suite "udp_unit" cargo test --lib net::udp -- --nocapture || true
run_suite "tcp_integration" cargo test --test net_tcp -- --nocapture || true
run_suite "udp_integration" cargo test --test net_udp -- --nocapture || true
run_suite "unix_integration" cargo test --test net_unix -- --nocapture || true
run_suite "net_hardening" cargo test --test net_hardening -- --nocapture || true
run_suite "net_verification" cargo test --test net_verification -- --nocapture || true

echo ""
echo ">>> [3/5] Analyzing logs for failure patterns..."
ISSUES=0

for pattern in "timed out" "connection refused" "broken pipe" "reset by peer"; do
    count=$(grep -rci "$pattern" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
    if [ "$count" -gt 0 ]; then
        echo "  NOTE: '$pattern' appeared $count time(s) (may be expected)"
    fi
done

if grep -rq "panicked at" "$LOG_DIR"/*.log 2>/dev/null; then
    echo "  WARNING: panics detected"
    grep -rh "panicked at" "$LOG_DIR"/*.log | head -5 > "${ARTIFACT_DIR}/panic_detected.txt" 2>/dev/null || true
    ISSUES=$((ISSUES + 1))
fi

if grep -rqi "leak" "$LOG_DIR"/*.log 2>/dev/null; then
    echo "  WARNING: potential leak detected"
    grep -rni "leak" "$LOG_DIR"/*.log | head -5 > "${ARTIFACT_DIR}/potential_leak.txt" 2>/dev/null || true
    ISSUES=$((ISSUES + 1))
fi

if grep -rqi "test result: FAILED" "$LOG_DIR"/*.log 2>/dev/null; then
    grep -rni "test result: FAILED" "$LOG_DIR"/*.log | head -5 > "${ARTIFACT_DIR}/cargo_reported_failures.txt" 2>/dev/null || true
fi

echo ""
echo ">>> [4/5] Collecting artifacts..."

PASSED_TESTS=$(grep -h -c "^test .* ok$" "$LOG_DIR"/*.log 2>/dev/null | awk '{s+=$1} END {print s+0}')
FAILED_TESTS=$(grep -h -c "^test .* FAILED$" "$LOG_DIR"/*.log 2>/dev/null | awk '{s+=$1} END {print s+0}')
RUN_ENDED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUITE_ID="net_hardening_e2e"
SCENARIO_ID="E2E-SUITE-NET-HARDENING"
REPRO_COMMAND="TEST_LOG_LEVEL=${TEST_LOG_LEVEL} RUST_LOG=${RUST_LOG} TEST_SEED=${TEST_SEED} SUITE_TIMEOUT=${SUITE_TIMEOUT} bash ${SCRIPT_DIR}/$(basename "$0")"
SUITE_STATUS="failed"
if [ "$FAILED_SUITES" -eq 0 ] && [ "$ISSUES" -eq 0 ]; then
    SUITE_STATUS="passed"
fi
FAILURE_CLASS="test_or_pattern_failure"
if [ "$SUITE_STATUS" = "passed" ]; then
    FAILURE_CLASS="none"
fi

cat > "${SUMMARY_JSON}" << ENDJSON
{
  "schema_version": "e2e-suite-summary-v3",
  "suite_id": "${SUITE_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "seed": "${TEST_SEED}",
  "started_ts": "${RUN_STARTED_TS}",
  "ended_ts": "${RUN_ENDED_TS}",
  "status": "${SUITE_STATUS}",
  "failure_class": "${FAILURE_CLASS}",
  "repro_command": "${REPRO_COMMAND}",
  "artifact_path": "${SUMMARY_JSON}",
  "suite": "${SUITE_ID}",
  "timestamp": "${TIMESTAMP}",
  "test_log_level": "${TEST_LOG_LEVEL}",
  "suite_timeout": ${SUITE_TIMEOUT},
  "suites_total": ${TOTAL_SUITES},
  "suites_passed": ${PASSED_SUITES},
  "suites_failed": ${FAILED_SUITES},
  "tests_passed": ${PASSED_TESTS},
  "tests_failed": ${FAILED_TESTS},
  "pattern_failures": ${ISSUES},
  "log_dir": "${LOG_DIR}",
  "artifact_dir": "${ARTIFACT_DIR}"
}
ENDJSON

grep -r -oE "seed[= ]+0x[0-9a-fA-F]+" "$LOG_DIR"/*.log > "${ARTIFACT_DIR}/seeds.txt" 2>/dev/null || true
grep -r -oE "trace_fingerprint[= ]+[a-f0-9]+" "$LOG_DIR"/*.log > "${ARTIFACT_DIR}/traces.txt" 2>/dev/null || true

cat > "${MANIFEST_FILE}" << ENDJSON
{
  "schema_version": "e2e-suite-artifact-manifest-v1",
  "suite_id": "${SUITE_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "summary_file": "${SUMMARY_JSON}",
  "summary_markdown": "${SUMMARY_MD}",
  "log_dir": "${LOG_DIR}",
  "artifact_dir": "${ARTIFACT_DIR}",
  "seed_file": "${ARTIFACT_DIR}/seeds.txt",
  "trace_file": "${ARTIFACT_DIR}/traces.txt"
}
ENDJSON

cat > "${SUMMARY_MD}" << EOF
# Network Hardening E2E Test Report

**Date:** $(date -Iseconds)

## Results

| Metric | Value |
|--------|-------|
| Suites Total | ${TOTAL_SUITES} |
| Suites Passed | ${PASSED_SUITES} |
| Suites Failed | ${FAILED_SUITES} |
| Pattern Issues | ${ISSUES} |

## Test Counts
$(grep -rh "^test result:" "$LOG_DIR"/*.log 2>/dev/null || echo "N/A")
EOF

echo "  Summary JSON: ${SUMMARY_JSON}"
echo "  Summary MD:   ${SUMMARY_MD}"
echo "  Manifest:     ${MANIFEST_FILE}"

echo ""
echo ">>> [5/5] Suite summary"
echo "==================================================================="
echo "  Seed:     ${TEST_SEED}"
echo "  Suites:   ${PASSED_SUITES}/${TOTAL_SUITES} passed"
echo "  Tests:    ${PASSED_TESTS} passed, ${FAILED_TESTS} failed"
echo "  Patterns: ${ISSUES} failure patterns"
echo "  Logs:     ${LOG_DIR}"
echo "  End:      $(date -Iseconds)"
echo "==================================================================="

find "$ARTIFACT_DIR" -name "*.txt" -empty -delete 2>/dev/null || true

if [ "$FAILED_SUITES" -gt 0 ] || [ "$ISSUES" -gt 0 ]; then
    exit 1
fi

echo ""
echo "All network hardening tests passed."
