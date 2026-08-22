#!/usr/bin/env python3

import json
from pathlib import Path

##############################################################################
# Configuration
##############################################################################

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT = PROJECT_ROOT / "data" / "pc-xp-llm-trace-full-theory-to-833.metta"
OUTPUT = PROJECT_ROOT / "data" / "trace_dataset.json"


##############################################################################
# Tokenizer
##############################################################################


def remove_comments(text: str) -> str:
    """Remove ;; comments."""
    lines = []

    for line in text.splitlines():
        line = line.split(";;", 1)[0]
        if line.strip():
            lines.append(line)

    return "\n".join(lines)


def tokenize(text):
    token = ""

    for c in text:
        if c == "(" or c == ")":
            if token:
                yield token
                token = ""
            yield c

        elif c.isspace():
            if token:
                yield token
                token = ""

        else:
            token += c

    if token:
        yield token


##############################################################################
# S-expression parser
##############################################################################


def parse(tokens):
    """
    Yields one top-level S-expression after another.

    Each expression becomes nested Python lists.
    """

    stack = []
    current = None

    for tok in tokens:
        if tok == "(":
            new = []

            if current is not None:
                stack.append(current)

            current = new

        elif tok == ")":
            if stack:
                finished = current
                current = stack.pop()
                current.append(finished)

            else:
                yield current
                current = None

        else:
            if current is None:
                print("Top-level token:", repr(tok))
                continue
            current.append(tok)


##############################################################################
# Main
##############################################################################

with INPUT.open("r", encoding="utf-8") as f:
    text = f.read()

text = remove_comments(text)

records = []

for expr in parse(tokenize(text)):
    if not expr:
        continue

    #
    # Ignore declarations such as
    #
    # (: TraceFullSmallest ...)
    #
    if expr[0] != "TraceFullSmallest":
        continue

    if len(expr) != 5:
        print("WARNING: unexpected TraceFullSmallest record:")
        print(expr)
        continue

    _, label, theory, goal, smallest = expr

    records.append(
        {
            "label": label,
            "goal_formula": goal,
            "theory": sorted(theory),  # order is irrelevant
            "smallest": sorted(smallest),  # order is irrelevant
        }
    )

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Loaded  {len(records)} theorem records.")
print(f"Written {OUTPUT}")
