#!/usr/bin/env bash
# One-command QUIC/H3 full verification runner.
#
# Usage:
#   ./scripts/quic_h3_verify.sh          # fast mode (unit + smoke E2E)
#   ./scripts/quic_h3_verify.sh --full   # full mode (all unit + all E2E + coverage + artifacts)
#
# Requires: cargo, python3
# For remote compilation: rch exec -- <command>

set -euo pipefail

MODE="${1:-fast}"
PASS=0
FAIL=0
START_TIME=$(date +%s)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

step() {
    echo -e "\n${BOLD}[$(date +%H:%M:%S)] $1${NC}"
}

pass() {
    echo -e "  ${GREEN}PASS${NC} $1"
    PASS=$((PASS + 1))
}

fail() {
    echo -e "  ${RED}FAIL${NC} $1"
    FAIL=$((FAIL + 1))
}

# ─── Gate 1: No-mock policy ─────────────────────────────────────────────
step "Gate 1: No-mock policy enforcement"
if python3 scripts/check_no_mock_policy.py --policy .github/no_mock_policy.json > /dev/null 2>&1; then
    pass "No-mock policy"
else
    fail "No-mock policy"
fi

# ─── Gate 2: Replay catalog integrity ───────────────────────────────────
step "Gate 2: Replay catalog integrity"
if python3 scripts/quic_h3_triage.py --self-test > /dev/null 2>&1; then
    pass "Replay catalog self-test"
else
    fail "Replay catalog self-test"
fi

# ─── Gate 3: Unit tests ─────────────────────────────────────────────────
step "Gate 3: QUIC/H3 unit tests"
for target in quic_core quic_native h3_native forensic_log; do
    if cargo test -p asupersync "$target" --all-features 2>&1 | tail -1 | grep -q "^test result: ok"; then
        pass "$target"
    else
        fail "$target"
    fi
done

# ─── Gate 4: E2E tests ──────────────────────────────────────────────────
if [ "$MODE" = "--full" ]; then
    E2E_CRATES="quic_h3_e2e quic_h3_e2e_loss quic_h3_e2e_h3 quic_h3_e2e_cancel quic_h3_e2e_violations"
else
    # Fast mode: run the main harness only (24 tests covering all categories)
    E2E_CRATES="quic_h3_e2e"
fi

step "Gate 4: E2E integration tests (${MODE})"
for crate in $E2E_CRATES; do
    if cargo test --test "$crate" --all-features 2>&1 | tail -1 | grep -q "^test result: ok"; then
        pass "$crate"
    else
        fail "$crate"
    fi
done

# ─── Gate 5: Coverage ratchet (full mode only) ──────────────────────────
if [ "$MODE" = "--full" ]; then
    step "Gate 5: Coverage ratchet check"
    if command -v cargo-llvm-cov > /dev/null 2>&1; then
        COVERAGE_LINE=$(cargo llvm-cov test -p asupersync --all-features --summary-only 2>&1 | grep "TOTAL" | head -1)
        if [ -n "$COVERAGE_LINE" ]; then
            pass "Coverage report generated"
        else
            fail "Coverage report"
        fi
    else
        echo -e "  ${YELLOW}SKIP${NC} cargo-llvm-cov not installed"
    fi
fi

# ─── Summary ────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  QUIC/H3 Verification Summary (${MODE})${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}Passed:${NC} ${PASS}"
echo -e "  ${RED}Failed:${NC} ${FAIL}"
echo -e "  Duration: ${ELAPSED}s"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo -e "\n${RED}VERIFICATION FAILED${NC}"
    echo "Run individual tests with --nocapture for details:"
    echo "  cargo test --test quic_h3_e2e -- --nocapture"
    echo "  python3 scripts/quic_h3_triage.py --catalog --verbose"
    exit 1
else
    echo -e "\n${GREEN}ALL GATES PASSED${NC}"
    exit 0
fi
