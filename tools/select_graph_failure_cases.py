from __future__ import annotations

import json
from pathlib import Path


REPORT = Path("outputs/graph_pilot/graph_builder_pilot_report.json")
OUT = Path("outputs/graph_pilot/graph_failure_cases.json")


def score_run(r: dict) -> float:
    score = 0.0

    score += min(r.get("component_count") or 0, 5000) / 100.0
    score += min(r.get("junction_count") or 0, 2000) / 50.0
    score += min(r.get("node_count") or 0, 10000) / 200.0

    for w in r.get("warnings", []):
        if w in {"too_many_components", "too_many_junctions", "too_many_short_branches"}:
            score += 20
        elif w in {"hkr_possible_form_grid", "hwr200_page", "large_page_scaled"}:
            score += 5
        else:
            score += 3

    if r["dataset"] in {"hwr200", "hkr_forms"}:
        score += 5

    return score


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    runs = data["runs"]

    ranked = sorted(
        [
            {
                **r,
                "failure_score": score_run(r),
            }
            for r in runs
        ],
        key=lambda x: x["failure_score"],
        reverse=True,
    )

    selected = ranked[:25]

    OUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {OUT}")
    print("top cases:")
    for r in selected[:15]:
        print(
            r["dataset"],
            r["method"],
            "score=", round(r["failure_score"], 2),
            "nodes=", r["node_count"],
            "components=", r["component_count"],
            "junctions=", r["junction_count"],
            "overlay=", r["overlay_path"],
        )


if __name__ == "__main__":
    main()