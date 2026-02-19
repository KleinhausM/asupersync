#!/usr/bin/env bash
set -euo pipefail

# Phase 6 End-to-End Test Runner
#
# Runs all five Phase 6 E2E suites plus a cross-component invariant flow
# and emits deterministic artifacts for CI/reporting.
#
# Usage:
#   ./scripts/run_phase6_e2e.sh              # run all suites
#   ./scripts/run_phase6_e2e.sh --suite geo  # run a single suite

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/target/phase6-e2e"
RCH_BIN="${RCH_BIN:-rch}"
PHASE6_TIMEOUT="${PHASE6_TIMEOUT:-1800}"
TEST_SEED="${TEST_SEED:-0xDEADBEEF}"
TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-info}"

mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STARTED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ARTIFACT_DIR="${OUTPUT_DIR}/artifacts_${TIMESTAMP}"
REPORT_FILE="${ARTIFACT_DIR}/report.txt"
SUITE_RESULTS_NDJSON="${ARTIFACT_DIR}/suite_results.ndjson"
COVERAGE_MAP_FILE="${ARTIFACT_DIR}/scenario_coverage_map.json"
REPLAY_POINTERS_FILE="${ARTIFACT_DIR}/replay_pointers.json"
SUMMARY_FILE="${ARTIFACT_DIR}/summary.json"
MASTER_LOG_FILE="${OUTPUT_DIR}/phase6_e2e_${TIMESTAMP}.log"

export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TEST_SEED
export TEST_LOG_LEVEL

if ! command -v "$RCH_BIN" >/dev/null 2>&1; then
    echo "Required executable not found: $RCH_BIN" >&2
    exit 1
fi

mkdir -p "$ARTIFACT_DIR"
: > "$SUITE_RESULTS_NDJSON"
: > "$REPORT_FILE"

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

json_bool() {
    if [[ "$1" -eq 1 ]]; then
        printf 'true'
    else
        printf 'false'
    fi
}

# Suite definitions: name, test target, required (1) or advisory (0)
declare -a SUITE_NAMES=(geo homo lyap raptorq plan)
declare -A SUITE_TARGETS=(
    [geo]=e2e_geodesic_normalization
    [homo]=topology_benchmark
    [lyap]=e2e_governor_vs_baseline
    [raptorq]=raptorq_conformance
    [plan]=golden_outputs
)
declare -A SUITE_LABELS=(
    [geo]="GEO  - Geodesic normalization"
    [homo]="HOMO - Topology-guided exploration"
    [lyap]="LYAP - Governor vs baseline"
    [raptorq]="RAPTORQ - Encode/decode conformance"
    [plan]="PLAN - Certified rewrite pipeline"
)
declare -A SUITE_SCENARIO_IDS=(
    [geo]="E2E-PHASE6-GEO"
    [homo]="E2E-PHASE6-HOMO"
    [lyap]="E2E-PHASE6-LYAP"
    [raptorq]="E2E-PHASE6-RAPTORQ"
    [plan]="E2E-PHASE6-PLAN"
)
declare -A SUITE_REPLAY_COMMANDS=(
    [geo]="RCH_BIN=${RCH_BIN} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} TEST_SEED=${TEST_SEED} ${RCH_BIN} exec -- cargo test --test e2e_geodesic_normalization --all-features -- --nocapture"
    [homo]="RCH_BIN=${RCH_BIN} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} TEST_SEED=${TEST_SEED} ${RCH_BIN} exec -- cargo test --test topology_benchmark --all-features -- --nocapture"
    [lyap]="RCH_BIN=${RCH_BIN} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} TEST_SEED=${TEST_SEED} ${RCH_BIN} exec -- cargo test --test e2e_governor_vs_baseline --all-features -- --nocapture"
    [raptorq]="NO_PREFLIGHT=1 TEST_SEED=${TEST_SEED} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} bash ${ROOT_DIR}/scripts/run_raptorq_e2e.sh --profile full"
    [plan]="RCH_BIN=${RCH_BIN} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} TEST_SEED=${TEST_SEED} ${RCH_BIN} exec -- cargo test --test golden_outputs --all-features -- --nocapture"
)

# Parse args
FILTER=""
if [[ "${1:-}" == "--suite" && -n "${2:-}" ]]; then
    FILTER="$2"
    if [[ -z "${SUITE_TARGETS[$FILTER]+x}" && "$FILTER" != "cross-component" ]]; then
        echo "Unknown suite: $FILTER"
        echo "Available: ${SUITE_NAMES[*]} cross-component"
        exit 1
    fi
fi

echo "==== Phase 6 End-to-End Test Suites ===="
echo "Output: ${REPORT_FILE}"
echo "Artifacts: ${ARTIFACT_DIR}"
echo ""

PASS=0
FAIL=0
TOTAL=0
INVARIANT_FLOW_FAILED=0

pushd "${ROOT_DIR}" >/dev/null

run_phase6_suite() {
    local name="$1"
    local target="$2"
    local label="$3"
    local scenario_id="$4"
    local replay_command="$5"
    local log_file="${ARTIFACT_DIR}/${name}.log"
    local rc=0

    printf "%-45s" "$label"
    TOTAL=$((TOTAL + 1))

    local started_epoch
    local ended_epoch
    local duration_ms
    started_epoch="$(date +%s)"

    set +e
    if [[ "$name" == "raptorq" ]]; then
        timeout "$PHASE6_TIMEOUT" bash "${ROOT_DIR}/scripts/run_raptorq_e2e.sh" --profile full > "$log_file" 2>&1
    else
        timeout "$PHASE6_TIMEOUT" "$RCH_BIN" exec -- cargo test --test "$target" --all-features -- --nocapture --test-threads=1 > "$log_file" 2>&1
    fi
    rc=$?
    set -e

    ended_epoch="$(date +%s)"
    duration_ms=$(((ended_epoch - started_epoch) * 1000))

    local passed
    local failed
    passed="$(grep -c "^test .* ok$" "$log_file" 2>/dev/null || true)"
    failed="$(grep -c "^test .* FAILED$" "$log_file" 2>/dev/null || true)"
    if [[ -z "$passed" ]]; then
        passed="0"
    fi
    if [[ -z "$failed" ]]; then
        failed="0"
    fi

    local status="failed"
    if [[ "$rc" -eq 0 ]]; then
        status="passed"
        echo "PASS  ($passed tests)"
        PASS=$((PASS + 1))
        echo "PASS  $label  ($passed tests)" >> "$REPORT_FILE"
    else
        echo "FAIL  ($passed passed, $failed failed)"
        FAIL=$((FAIL + 1))
        echo "FAIL  $label  ($passed passed, $failed failed, exit=$rc)" >> "$REPORT_FILE"
        echo "  Log: $log_file" >> "$REPORT_FILE"
    fi

    printf '{"schema_version":"phase6-suite-result-v1","suite":"%s","suite_id":"phase6_%s","scenario_id":"%s","status":"%s","exit_code":%d,"duration_ms":%d,"tests_passed":%d,"tests_failed":%d,"log_file":"%s","replay_command":"%s"}\n' \
        "$(json_escape "$name")" \
        "$(json_escape "$name")" \
        "$(json_escape "$scenario_id")" \
        "$(json_escape "$status")" \
        "$rc" \
        "$duration_ms" \
        "$passed" \
        "$failed" \
        "$(json_escape "$log_file")" \
        "$(json_escape "$replay_command")" >> "$SUITE_RESULTS_NDJSON"
}

for name in "${SUITE_NAMES[@]}"; do
    if [[ -n "$FILTER" && "$name" != "$FILTER" ]]; then
        continue
    fi

    target="${SUITE_TARGETS[$name]}"
    label="${SUITE_LABELS[$name]}"
    scenario_id="${SUITE_SCENARIO_IDS[$name]}"
    replay_command="${SUITE_REPLAY_COMMANDS[$name]}"
    run_phase6_suite "$name" "$target" "$label" "$scenario_id" "$replay_command"
done

# Cross-component flow to validate loser-drain and region-quiescence end-to-end.
if [[ -z "$FILTER" || "$FILTER" == "cross-component" ]]; then
    echo ""
    echo "Cross-component invariant flow (runtime_e2e):"
    TOTAL=$((TOTAL + 1))
    flow_log="${ARTIFACT_DIR}/cross_component_invariant_flow.log"
    flow_replay="RCH_BIN=${RCH_BIN} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} TEST_SEED=${TEST_SEED} bash ${ROOT_DIR}/scripts/run_phase6_e2e.sh --suite cross-component"
    flow_start="$(date +%s)"
    flow_rc=0

    set +e
    {
        timeout "$PHASE6_TIMEOUT" "$RCH_BIN" exec -- cargo test --test runtime_e2e e2e_task_spawn_and_quiescence -- --nocapture --test-threads=1
        rc_quiescence=$?
        timeout "$PHASE6_TIMEOUT" "$RCH_BIN" exec -- cargo test --test runtime_e2e e2e_race_loser_drain -- --nocapture --test-threads=1
        rc_loser=$?
        if [[ "$rc_quiescence" -ne 0 || "$rc_loser" -ne 0 ]]; then
            exit 1
        fi
        exit 0
    } > "$flow_log" 2>&1
    flow_rc=$?
    set -e

    flow_end="$(date +%s)"
    flow_duration_ms=$(((flow_end - flow_start) * 1000))
    flow_passed="$(grep -c "^test .* ok$" "$flow_log" 2>/dev/null || true)"
    flow_failed="$(grep -c "^test .* FAILED$" "$flow_log" 2>/dev/null || true)"
    if [[ -z "$flow_passed" ]]; then
        flow_passed="0"
    fi
    if [[ -z "$flow_failed" ]]; then
        flow_failed="0"
    fi
    flow_status="failed"
    if [[ "$flow_rc" -eq 0 ]]; then
        flow_status="passed"
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        INVARIANT_FLOW_FAILED=1
    fi

    echo "  status: ${flow_status} (log: ${flow_log})"
    printf '{"schema_version":"phase6-suite-result-v1","suite":"cross-component","suite_id":"phase6_cross_component","scenario_id":"E2E-PHASE6-CROSS-COMPONENT","status":"%s","exit_code":%d,"duration_ms":%d,"tests_passed":%d,"tests_failed":%d,"log_file":"%s","replay_command":"%s","invariants":["loser_drain","region_quiescence"]}\n' \
        "$(json_escape "$flow_status")" \
        "$flow_rc" \
        "$flow_duration_ms" \
        "$flow_passed" \
        "$flow_failed" \
        "$(json_escape "$flow_log")" \
        "$(json_escape "$flow_replay")" >> "$SUITE_RESULTS_NDJSON"
fi

popd >/dev/null

cat "${ARTIFACT_DIR}"/*.log > "$MASTER_LOG_FILE" 2>/dev/null || true

jq -s '{
  schema_version: "phase6-scenario-coverage-v1",
  suite_id: "phase6_e2e",
  generated_at: "'"${TIMESTAMP}"'",
  entries: map({
    scenario_id: .scenario_id,
    suite: .suite,
    status: .status,
    log_file: .log_file
  })
}' "$SUITE_RESULTS_NDJSON" > "$COVERAGE_MAP_FILE"

jq -s '{
  schema_version: "phase6-replay-pointers-v1",
  suite_id: "phase6_e2e",
  generated_at: "'"${TIMESTAMP}"'",
  pointers: map({
    scenario_id: .scenario_id,
    suite: .suite,
    replay_command: .replay_command
  })
}' "$SUITE_RESULTS_NDJSON" > "$REPLAY_POINTERS_FILE"

RUN_ENDED_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUITE_STATUS="failed"
if [[ "$FAIL" -eq 0 && "$INVARIANT_FLOW_FAILED" -eq 0 ]]; then
    SUITE_STATUS="passed"
fi

FAILURE_CLASS="phase6_suite_failure"
if [[ "$INVARIANT_FLOW_FAILED" -eq 1 ]]; then
    FAILURE_CLASS="cross_component_invariant_failure"
elif [[ "$SUITE_STATUS" == "passed" ]]; then
    FAILURE_CLASS="none"
fi

REPRO_COMMAND="RCH_BIN=${RCH_BIN} PHASE6_TIMEOUT=${PHASE6_TIMEOUT} TEST_SEED=${TEST_SEED} bash ${ROOT_DIR}/scripts/run_phase6_e2e.sh"

cat > "$SUMMARY_FILE" << ENDJSON
{
  "schema_version": "e2e-suite-summary-v3",
  "suite_id": "phase6_e2e",
  "scenario_id": "E2E-SUITE-PHASE6",
  "seed": "${TEST_SEED}",
  "started_ts": "${RUN_STARTED_TS}",
  "ended_ts": "${RUN_ENDED_TS}",
  "status": "${SUITE_STATUS}",
  "failure_class": "${FAILURE_CLASS}",
  "repro_command": "${REPRO_COMMAND}",
  "artifact_path": "${SUMMARY_FILE}",
  "suite": "phase6_e2e",
  "timestamp": "${TIMESTAMP}",
  "test_log_level": "${TEST_LOG_LEVEL}",
  "total_suites": ${TOTAL},
  "passed_suites": ${PASS},
  "failed_suites": ${FAIL},
  "cross_component_invariant_failed": $(json_bool "$INVARIANT_FLOW_FAILED"),
  "report_file": "${REPORT_FILE}",
  "suite_results_ndjson": "${SUITE_RESULTS_NDJSON}",
  "scenario_coverage_map": "${COVERAGE_MAP_FILE}",
  "replay_pointers": "${REPLAY_POINTERS_FILE}",
  "log_file": "${MASTER_LOG_FILE}",
  "artifact_dir": "${ARTIFACT_DIR}"
}
ENDJSON

echo ""
echo "---- Summary ----"
echo "Suites: $TOTAL  Pass: $PASS  Fail: $FAIL"
echo "Cross-component invariant flow failed: $INVARIANT_FLOW_FAILED"
echo "Report: ${REPORT_FILE}"
echo "Summary: ${SUMMARY_FILE}"
echo "Coverage map: ${COVERAGE_MAP_FILE}"
echo "Replay pointers: ${REPLAY_POINTERS_FILE}"
echo "Logs:   ${ARTIFACT_DIR}/"

echo "" >> "$REPORT_FILE"
echo "Summary: $TOTAL suites, $PASS passed, $FAIL failed" >> "$REPORT_FILE"
echo "Cross-component invariant flow failed: $INVARIANT_FLOW_FAILED" >> "$REPORT_FILE"
echo "Timestamp: $TIMESTAMP" >> "$REPORT_FILE"

if [[ "$SUITE_STATUS" != "passed" ]]; then
    exit 1
fi
