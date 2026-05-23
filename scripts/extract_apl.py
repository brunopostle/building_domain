"""Extract Alexander's 253 patterns from apl.html into structured JSON.

Usage:
    python scripts/extract_apl.py [--input PATH] [--output PATH]

Defaults:
    input:  /home/bruno/src/apl/apl.html
    output: data/apl_patterns.json
"""
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup


DEFAULT_INPUT = Path("/home/bruno/src/apl/apl.html")
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "apl_patterns.json"


def parse_pattern_ref(link_tag) -> dict:
    """Parse a pattern link tag into {number, name, confidence, id}."""
    href = link_tag.get("href", "")
    slug = href.lstrip("#")
    text = link_tag.get_text(strip=True)

    confidence = ""
    if text.endswith("**"):
        confidence = "**"
        text = text[:-2].strip()
    elif text.endswith("*"):
        confidence = "*"
        text = text[:-1].strip()

    # first token is the number
    parts = text.split(None, 1)
    number = int(parts[0]) if parts and parts[0].isdigit() else None
    name = parts[1].strip() if len(parts) > 1 else text

    return {"number": number, "name": name, "confidence": confidence, "id": slug}


def collect_text(tags) -> str:
    """Concatenate text from a list of tags, joining with newlines."""
    parts = []
    for tag in tags:
        t = tag.get_text(separator=" ", strip=True)
        if t:
            parts.append(t)
    return "\n".join(parts)


def extract_patterns(html_path: Path) -> list[dict]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    patterns = []
    current_section = ""

    for element in soup.body.children:
        if not hasattr(element, "name") or element.name is None:
            continue

        if element.name == "h2":
            current_section = element.get_text(strip=True)
            continue

        if element.name != "details" or "pattern" not in element.get("class", []):
            continue

        slug = element.get("id", "")

        # --- number, name, confidence from summary ---
        summary = element.find("summary")
        summary_text = summary.get_text(strip=True) if summary else ""

        confidence = ""
        if summary_text.endswith("**"):
            confidence = "**"
            summary_text = summary_text[:-2].strip()
        elif summary_text.endswith("*"):
            confidence = "*"
            summary_text = summary_text[:-1].strip()

        parts = summary_text.split(None, 1)
        number = int(parts[0]) if parts and parts[0].isdigit() else None
        name = parts[1].strip() if len(parts) > 1 else summary_text

        # --- problem, solution, higher/lower patterns from content ---
        content = element.find("div", class_="pattern-content")
        if not content:
            continue

        problem_paras = []
        solution_paras = []
        higher_patterns = []
        lower_patterns = []

        mode = None  # "problem" | "solution"

        for p in content.find_all("p", recursive=False):
            classes = p.get("class", [])

            if "higher-patterns" in classes:
                for a in p.find_all("a"):
                    higher_patterns.append(parse_pattern_ref(a))
                continue

            if "lower-patterns" in classes:
                for a in p.find_all("a"):
                    lower_patterns.append(parse_pattern_ref(a))
                continue

            strong = p.find("strong")
            if strong:
                label = strong.get_text(strip=True).rstrip(":").lower()
                if label == "problem":
                    mode = "problem"
                    # strip the "Problem:" strong from the text
                    strong.decompose()
                    problem_paras.append(p)
                    continue
                elif label == "solution":
                    mode = "solution"
                    strong.decompose()
                    solution_paras.append(p)
                    continue

            if mode == "problem":
                problem_paras.append(p)
            elif mode == "solution":
                solution_paras.append(p)

        patterns.append({
            "number": number,
            "id": slug,
            "name": name,
            "confidence": confidence,
            "section": current_section,
            "problem": collect_text(problem_paras),
            "solution": collect_text(solution_paras),
            "higher_patterns": higher_patterns,
            "lower_patterns": lower_patterns,
        })

    return patterns


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    patterns = extract_patterns(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(patterns, indent=2, ensure_ascii=False))

    print(f"Extracted {len(patterns)} patterns → {args.output}")

    # quick sanity check
    missing_problem = [p for p in patterns if not p["problem"]]
    missing_solution = [p for p in patterns if not p["solution"]]
    if missing_problem:
        print(f"WARNING: {len(missing_problem)} patterns have no problem text: "
              f"{[p['number'] for p in missing_problem]}")
    if missing_solution:
        print(f"WARNING: {len(missing_solution)} patterns have no solution text: "
              f"{[p['number'] for p in missing_solution]}")


if __name__ == "__main__":
    main()
