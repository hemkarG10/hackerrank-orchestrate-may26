# Support Triage Agent

This directory contains a deterministic, terminal-based support triage agent for
the HackerRank Orchestrate challenge.

## Approach

The agent uses only the local corpus in `../data/`.

1. Load all Markdown support articles from HackerRank, Claude, and Visa.
2. Build a small BM25-style retrieval index without network calls or external
   packages.
3. Route sensitive or unsupported tickets through explicit safety rules.
4. Use the retrieved support article as evidence for the response and
   justification.
5. Validate the final structured CSV fields before writing output.

This design is intentionally deterministic and auditable. It prioritizes
grounded answers, correct escalation, and reproducible CSV output over a
free-form chatbot loop.

## Run

From the repository root:

```bash
python3 code/main.py
```

By default this reads:

```text
support_tickets/support_tickets.csv
```

and writes:

```text
support_tickets/output.csv
```

You can also pass explicit paths:

```bash
python3 code/main.py \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv \
  --data data
```

## Dependencies

No third-party Python packages are required. The code uses only the Python
standard library.

## Output Schema

The generated CSV contains:

```text
issue,subject,company,response,product_area,status,request_type,justification
```

Allowed values are enforced for:

- `status`: `replied`, `escalated`
- `request_type`: `product_issue`, `feature_request`, `bug`, `invalid`

## Notes For Judges

- The agent does not call live web services.
- Secrets are not required and no API keys are read.
- Account-specific actions such as refunds, access restoration, score changes,
  and broad service incidents are escalated instead of guessed.
