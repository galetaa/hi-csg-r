from __future__ import annotations

import json
from pathlib import Path


REPORT = Path("outputs/graph_pilot_v2/graph_builder_pilot_report.json")
OUT = Path("outputs/graph_pilot_v2/graph_failure_cases_v2.json")


def score_run(r: dict) -> float:
    score = 0.0

    dataset = r["dataset"]
    method = r["method"]

    score += min(r.get("component_count") or 0, 8000) / 80.0
    score += min(r.get("junction_count") or 0, 4000) / 80.0
    score += min(r.get("node_count") or 0, 15000) / 300.0

    for w in r.get("warnings", []):
        if w in {"too_many_components", "too_many_junctions", "too_many_short_branches"}:
            score += 20
        elif w in {"too_high_foreground_ratio", "no_special_nodes_detected"}:
            score += 18
        elif w in {"hkr_possible_form_grid", "hwr200_page", "large_page_scaled"}:
            score += 6
        else:
            score += 3

    if dataset in {"hwr200", "hkr_forms"}:
        score += 10

    if dataset == "school_notebooks" and method == "otsu":
        score += 8

    return score


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    runs = data["runs"]

    ranked = sorted(
        [{**r, "failure_score": score_run(r)} for r in runs],
        key=lambda x: x["failure_score"],
        reverse=True,
    )

    selected = ranked[:40]

    OUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", OUT)
    print("top cases:")
    for r in selected[:25]:
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