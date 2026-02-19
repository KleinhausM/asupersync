#!/usr/bin/env python3
"""Generate D5 CI summary artifacts (machine-readable + markdown)."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_iso8601_utc(raw: str) -> dt.datetime:
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {raw}")
    return parsed.astimezone(dt.timezone.utc)


def route_owner(path: str, routes: list[dict[str, Any]], default_owner: str) -> str:
    for route in routes:
        pattern = route.get("pattern")
        owner = route.get("owner")
        if isinstance(pattern, str) and isinstance(owner, str) and fnmatch.fnmatch(path, pattern):
            return owner
    return default_owner


def run_scan(roots: list[str], terms: list[str]) -> dict[str, list[dict[str, Any]]]:
    escaped_terms = [re.escape(term) for term in terms]
    token_re = re.compile(rf"(?i)\b({'|'.join(escaped_terms)})\b")

    cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
    for term in terms:
        cmd.extend(["-e", rf"(?i)\b{re.escape(term)}\b"])
    cmd.extend(roots)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise RuntimeError("ripgrep scan failed")
    if proc.returncode == 1:
        return {}

    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in proc.stdout.splitlines():
        parts = row.split(":", 2)
        if len(parts) != 3:
            continue
        path, line_raw, text = parts
        try:
            line = int(line_raw)
        except ValueError:
            continue
        tokens = sorted({m.group(1).lower() for m in token_re.finditer(text)})
        if not tokens:
            continue
        by_path.setdefault(path, []).append(
            {
                "line": line,
                "tokens": tokens,
                "text": text,
            }
        )
    return by_path


def no_mock_snapshot(policy_path: Path, output: Path) -> int:
    policy = load_json(policy_path)
    if policy.get("schema_version") != "no-mock-policy-v1":
        raise ValueError("unsupported no-mock policy schema")

    roots = list(policy.get("scan", {}).get("roots", ["src", "tests"]))
    terms = list(policy.get("scan", {}).get("terms", ["mock", "fake", "stub"]))
    allowlist = set(policy.get("allowlist_paths", []))
    waivers = list(policy.get("waivers", []))
    routes = list(policy.get("owner_routes", []))
    default_owner = str(policy.get("default_owner", "runtime-core"))
    now_utc = dt.datetime.now(dt.timezone.utc)

    active_waiver_by_path: dict[str, dict[str, Any]] = {}
    expired_waivers: list[dict[str, Any]] = []
    for waiver in waivers:
        path = waiver.get("path")
        status = waiver.get("status")
        expiry_raw = waiver.get("expires_at_utc")
        if not isinstance(path, str) or not isinstance(status, str) or not isinstance(expiry_raw, str):
            continue
        expiry = parse_iso8601_utc(expiry_raw)
        if status == "active":
            active_waiver_by_path[path] = waiver
            if expiry <= now_utc:
                expired_waivers.append(waiver)

    hits_by_path = run_scan(roots, terms)

    violations: list[dict[str, Any]] = []
    for path, path_hits in sorted(hits_by_path.items()):
        if path in allowlist:
            continue
        waiver = active_waiver_by_path.get(path)
        if waiver is not None and waiver not in expired_waivers:
            continue
        owner = route_owner(path, routes, default_owner)
        tokens = sorted({token for hit in path_hits for token in hit["tokens"]})
        first_line = min(hit["line"] for hit in path_hits)
        violations.append(
            {
                "path": path,
                "owner": owner,
                "first_line": first_line,
                "tokens": tokens,
                "hit_count": len(path_hits),
            }
        )

    report = {
        "schema_version": "no-mock-policy-report-v1",
        "generated_at": utc_now(),
        "policy_path": str(policy_path),
        "scan": {
            "roots": roots,
            "terms": terms,
        },
        "policy_counts": {
            "allowlist_paths": len(allowlist),
            "waivers_total": len(waivers),
            "waivers_active": sum(1 for waiver in waivers if waiver.get("status") == "active"),
        },
        "scan_counts": {
            "matching_paths": len(hits_by_path),
            "matching_hits": sum(len(path_hits) for path_hits in hits_by_path.values()),
            "violating_paths": len(violations),
            "expired_waivers": len(expired_waivers),
        },
        "expired_waivers": [
            {
                "waiver_id": waiver.get("waiver_id", "<unknown>"),
                "path": waiver.get("path"),
                "owner": waiver.get("owner", route_owner(str(waiver.get("path", "")), routes, default_owner)),
                "expires_at_utc": waiver.get("expires_at_utc"),
                "replacement_issue": waiver.get("replacement_issue"),
            }
            for waiver in expired_waivers
        ],
        "violations": violations,
        "status": "pass" if not violations and not expired_waivers else "fail",
    }

    write_json(output, report)
    print(f"No-mock policy report: {output}")
    return 0


def read_report(path: Path, required_schema: str, label: str) -> dict[str, Any]:
    payload = load_json(path)
    schema = payload.get("schema_version")
    if schema != required_schema:
        raise ValueError(f"{label} schema mismatch: expected {required_schema}, got {schema}")
    return payload


def read_previous(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = load_json(path)
    if payload.get("schema_version") != "ci-summary-report-v1":
        return None
    return payload


def nested_get(payload: dict[str, Any], keys: list[str]) -> float | int | None:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, (int, float)):
        return cursor
    return None


def delta(current: dict[str, Any], previous: dict[str, Any] | None, keys: list[str]) -> float | None:
    if previous is None:
        return None
    current_value = nested_get(current, keys)
    previous_value = nested_get(previous, keys)
    if current_value is None or previous_value is None:
        return None
    return round(float(current_value) - float(previous_value), 4)


def compose_summary(
    coverage_report: Path,
    no_mock_report: Path,
    e2e_matrix_report: Path,
    forensics_report: Path,
    output_json: Path,
    output_markdown: Path,
    previous_report: Path | None,
    fail_on_nonpass: bool,
) -> int:
    coverage = read_report(coverage_report, "coverage-ratchet-report-v1", "coverage report")
    no_mock = read_report(no_mock_report, "no-mock-policy-report-v1", "no-mock report")
    e2e_matrix = read_report(e2e_matrix_report, "e2e-scenario-matrix-validation-v1", "e2e matrix report")
    forensics = read_report(forensics_report, "raptorq-e2e-suite-log-v1", "forensics report")
    previous = read_previous(previous_report)

    coverage_section = {
        "status": coverage.get("status", "fail"),
        "global_line_pct": coverage.get("global_coverage", {}).get("line_pct"),
        "global_floor_pct": coverage.get("global_coverage", {}).get("floor_pct"),
        "failure_count": coverage.get("failure_count", 0),
        "failing_subsystems": [
            row.get("id")
            for row in coverage.get("subsystem_results", [])
            if row.get("status") != "pass"
        ],
    }
    no_mock_section = {
        "status": no_mock.get("status", "fail"),
        "matching_paths": no_mock.get("scan_counts", {}).get("matching_paths", 0),
        "violating_paths": no_mock.get("scan_counts", {}).get("violating_paths", 0),
        "expired_waivers": no_mock.get("scan_counts", {}).get("expired_waivers", 0),
        "allowlist_paths": no_mock.get("policy_counts", {}).get("allowlist_paths", 0),
        "active_waivers": no_mock.get("policy_counts", {}).get("waivers_active", 0),
    }
    e2e_matrix_section = {
        "status": e2e_matrix.get("status", "fail"),
        "suite_row_count": e2e_matrix.get("suite_row_count", 0),
        "raptorq_row_count": e2e_matrix.get("raptorq_row_count", 0),
        "suite_failures": e2e_matrix.get("suite_failures", 0),
        "raptorq_failures": e2e_matrix.get("raptorq_failures", 0),
    }
    forensics_section = {
        "status": forensics.get("status", "fail"),
        "profile": forensics.get("profile"),
        "selected_scenarios": forensics.get("selected_scenarios", 0),
        "passed_scenarios": forensics.get("passed_scenarios", 0),
        "failed_scenarios": forensics.get("failed_scenarios", 0),
    }

    sections = {
        "coverage": coverage_section,
        "no_mock": no_mock_section,
        "e2e_matrix": e2e_matrix_section,
        "forensics": forensics_section,
    }
    overall_status = "pass" if all(section.get("status") == "pass" for section in sections.values()) else "fail"

    report = {
        "schema_version": "ci-summary-report-v1",
        "generated_at": utc_now(),
        "overall_status": overall_status,
        "sources": {
            "coverage_report": str(coverage_report),
            "no_mock_report": str(no_mock_report),
            "e2e_matrix_report": str(e2e_matrix_report),
            "forensics_report": str(forensics_report),
            "previous_report": str(previous_report) if previous_report else None,
        },
        "sections": sections,
        "trends": {
            "coverage_global_line_pct_delta": delta(
                {"sections": sections}, previous, ["sections", "coverage", "global_line_pct"]
            ),
            "no_mock_matching_paths_delta": delta(
                {"sections": sections}, previous, ["sections", "no_mock", "matching_paths"]
            ),
            "no_mock_violating_paths_delta": delta(
                {"sections": sections}, previous, ["sections", "no_mock", "violating_paths"]
            ),
            "e2e_matrix_failures_delta": delta(
                {"sections": sections}, previous, ["sections", "e2e_matrix", "suite_failures"]
            ),
            "forensics_failed_scenarios_delta": delta(
                {"sections": sections}, previous, ["sections", "forensics", "failed_scenarios"]
            ),
        },
    }

    write_json(output_json, report)

    md_lines = [
        "# CI Summary Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Overall status: `{overall_status}`",
        "",
        "| Area | Status | Key Metrics | Trend |",
        "| --- | --- | --- | --- |",
        (
            "| Coverage | "
            f"`{coverage_section['status']}` | "
            f"global={coverage_section['global_line_pct']}% floor={coverage_section['global_floor_pct']}% "
            f"failures={coverage_section['failure_count']} | "
            f"delta_global={report['trends']['coverage_global_line_pct_delta']} |"
        ),
        (
            "| No-mock policy | "
            f"`{no_mock_section['status']}` | "
            f"matches={no_mock_section['matching_paths']} violations={no_mock_section['violating_paths']} "
            f"expired_waivers={no_mock_section['expired_waivers']} | "
            f"delta_violations={report['trends']['no_mock_violating_paths_delta']} |"
        ),
        (
            "| E2E matrix (D4) | "
            f"`{e2e_matrix_section['status']}` | "
            f"suite_failures={e2e_matrix_section['suite_failures']} "
            f"raptorq_failures={e2e_matrix_section['raptorq_failures']} | "
            f"delta_suite_failures={report['trends']['e2e_matrix_failures_delta']} |"
        ),
        (
            "| Forensics (D3) | "
            f"`{forensics_section['status']}` | "
            f"profile={forensics_section['profile']} selected={forensics_section['selected_scenarios']} "
            f"failed={forensics_section['failed_scenarios']} | "
            f"delta_failed={report['trends']['forensics_failed_scenarios_delta']} |"
        ),
        "",
        "## Notes",
        f"- Coverage failing subsystems: `{coverage_section['failing_subsystems']}`",
        "- Trend deltas are `None` when no previous D5 summary artifact was provided.",
    ]
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"CI summary JSON: {output_json}")
    print(f"CI summary Markdown: {output_markdown}")

    if fail_on_nonpass and overall_status != "pass":
        print("CI summary status is non-pass")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    no_mock_parser = subparsers.add_parser(
        "no-mock-snapshot",
        help="Generate machine-readable no-mock policy snapshot.",
    )
    no_mock_parser.add_argument("--policy", required=True, type=Path)
    no_mock_parser.add_argument("--output", required=True, type=Path)

    compose_parser = subparsers.add_parser(
        "compose",
        help="Compose D5 machine-readable + markdown summary from CI artifacts.",
    )
    compose_parser.add_argument("--coverage-report", required=True, type=Path)
    compose_parser.add_argument("--no-mock-report", required=True, type=Path)
    compose_parser.add_argument("--e2e-matrix-report", required=True, type=Path)
    compose_parser.add_argument("--forensics-report", required=True, type=Path)
    compose_parser.add_argument("--output-json", required=True, type=Path)
    compose_parser.add_argument("--output-markdown", required=True, type=Path)
    compose_parser.add_argument("--previous-report", type=Path)
    compose_parser.add_argument("--fail-on-nonpass", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "no-mock-snapshot":
        return no_mock_snapshot(policy_path=args.policy, output=args.output)
    if args.command == "compose":
        return compose_summary(
            coverage_report=args.coverage_report,
            no_mock_report=args.no_mock_report,
            e2e_matrix_report=args.e2e_matrix_report,
            forensics_report=args.forensics_report,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            previous_report=args.previous_report,
            fail_on_nonpass=args.fail_on_nonpass,
        )
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
