#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible LLM (e.g. DeepSeek) against the Pass 3 schema.

Verifies, before committing to a full 12-pass run, that:
  1. Auth + base URL are correct (a real call succeeds).
  2. instructor's tool-calling / structured-output mode validates the
     AssertionExtractionResponse schema cleanly against the target model.
  3. Assertion DENSITY is comparable to Haiku (~10-20 per entity), NOT the
     sparse 1-2 seen with Groq/Llama-70B, which makes the model unusable for
     Pass 3+.

This reuses the production OpenAIProvider, the real Pass 3 framing template, and
the real AssertionExtractionResponse schema, so a pass here means the actual
pipeline will validate too.

Usage (in a separate terminal, NOT inside a Claude Code session):

    export OPENAI_API_KEY=<deepseek key>          # starts with sk-...
    export OPENAI_BASE_URL=https://api.deepseek.com
    python scripts/smoke_test_deepseek.py                 # default: deepseek-chat
    python scripts/smoke_test_deepseek.py --model deepseek-chat

Exit code 0 = density looks healthy (mean >= 5/entity); 1 = config/auth error
or sparse output (investigate before running the full pipeline).
"""
import argparse
import os
import sys

# A handful of common, well-understood building concepts. Each should produce a
# rich set of relationships from any competent model.
SAMPLE_ENTITIES = [
    ("Window", "component"),
    ("External Wall", "component"),
    ("Flat Roof", "component"),
    ("Strip Foundation", "component"),
    ("Underfloor Heating", "system"),
]

SPARSE_THRESHOLD = 5.0  # mean assertions/entity below this => treat as Groq-like sparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-chat", help="OpenAI-compatible model id")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "<default OpenAI>")
    if "deepseek" in args.model and "deepseek" not in base_url:
        print(
            f"WARNING: model '{args.model}' but OPENAI_BASE_URL={base_url!r} — "
            "this is probably a misconfiguration (a Groq key won't work against "
            "DeepSeek, and vice versa).",
            file=sys.stderr,
        )

    # Import after env check so the error message above is the first thing seen.
    from bsos.llm.openai_provider import OpenAIProvider
    from bsos.pipeline.pass3 import FRAMING_TEMPLATES
    from bsos.pipeline.schemas import AssertionExtractionResponse

    # cache=None so we always hit the live API.
    provider = OpenAIProvider(args.model)

    print(f"Model:    {args.model}")
    print(f"Base URL: {base_url}")
    print(f"Schema:   AssertionExtractionResponse (Pass 3, framing 1)\n")

    template = FRAMING_TEMPLATES[0]
    counts: list[int] = []
    for name, etype in SAMPLE_ENTITIES:
        prompt = template.format(name=name, entity_type=etype)
        try:
            response = provider.extract(prompt, AssertionExtractionResponse, entity_name=name)
        except SystemExit as exc:  # provider sys.exit()s on 400/401/403
            print(f"\nFATAL: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            msg = str(exc)
            # instructor wraps a 401 in InstructorRetryException, so the provider's
            # APIStatusError fast-path never fires. Detect auth failures here and
            # abort immediately rather than retrying through every sample entity.
            if "401" in msg or "authentication" in msg.lower():
                print(
                    f"\nFATAL: authentication failed for model {args.model} at {base_url}. "
                    "Check OPENAI_API_KEY matches the provider for OPENAI_BASE_URL.",
                    file=sys.stderr,
                )
                return 1
            print(f"  {name:<20} SCHEMA/CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            counts.append(0)
            continue

        n = len(response.assertions)
        counts.append(n)
        sample = response.assertions[0] if response.assertions else None
        sample_str = f"  e.g. {name} {sample.predicate} {sample.object_name}" if sample else ""
        print(f"  {name:<20} {n:>3} assertions{sample_str}")

    mean = sum(counts) / len(counts) if counts else 0.0
    print(f"\nMean assertions/entity: {mean:.1f}")

    if mean < SPARSE_THRESHOLD:
        print(
            f"SPARSE ({mean:.1f} < {SPARSE_THRESHOLD}): output resembles Groq/Llama-70B. "
            "Not suitable for Pass 3+. Do NOT run the full pipeline with this model.",
            file=sys.stderr,
        )
        return 1

    print(f"HEALTHY: density comparable to Haiku. Safe to run the full pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
