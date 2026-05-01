#!/usr/bin/env python3
"""Terminal entry point for the HackerRank Orchestrate support agent."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from agent import TriageAgent
from corpus import CorpusIndex


OUTPUT_FIELDS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def row_get(row: dict[str, str], key: str) -> str:
    for candidate in (key, key.lower(), key.upper(), key.title()):
        if candidate in row:
            return row[candidate] or ""
    return ""


def read_tickets(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_predictions(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    allowed_statuses = {"replied", "escalated"}
    allowed_types = {"product_issue", "feature_request", "bug", "invalid"}
    for index, row in enumerate(rows, start=1):
        for field in OUTPUT_FIELDS:
            if field not in row:
                errors.append(f"row {index}: missing field {field}")
        if row.get("status") not in allowed_statuses:
            errors.append(f"row {index}: invalid status {row.get('status')!r}")
        if row.get("request_type") not in allowed_types:
            errors.append(f"row {index}: invalid request_type {row.get('request_type')!r}")
        for field in ("response", "status", "request_type", "justification"):
            if not (row.get(field) or "").strip():
                errors.append(f"row {index}: blank {field}")
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Resolve support tickets using the local support corpus only."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "support_tickets" / "support_tickets.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "support_tickets" / "output.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data",
        help="Local support corpus directory.",
    )
    parser.add_argument(
        "--show-low-confidence",
        action="store_true",
        help="Print rows that fell back to generic retrieval.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    tickets = read_tickets(args.input)
    index = CorpusIndex.from_directory(args.data)
    agent = TriageAgent(index)

    predictions: list[dict[str, str]] = []
    low_confidence: list[tuple[int, str, str]] = []
    for i, ticket in enumerate(tickets, start=1):
        issue = row_get(ticket, "issue")
        subject = row_get(ticket, "subject")
        company = row_get(ticket, "company")
        decision = agent.resolve(issue=issue, subject=subject, company=company)
        predictions.append(
            {
                "issue": issue,
                "subject": subject,
                "company": company,
                "response": decision.response,
                "product_area": decision.product_area,
                "status": decision.status,
                "request_type": decision.request_type,
                "justification": decision.justification,
            }
        )
        if decision.rule_name == "retrieval_fallback":
            low_confidence.append((i, subject, company))

    errors = validate_predictions(predictions)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    write_predictions(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    if args.show_low_confidence and low_confidence:
        print("Rows handled by retrieval fallback:")
        for row_number, subject, company in low_confidence:
            print(f"- row {row_number}: [{company}] {subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
