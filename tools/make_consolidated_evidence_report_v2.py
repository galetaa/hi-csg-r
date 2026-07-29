from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print("wrote:", path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote:", path)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def signed(value: Any, digits: int = 5) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def pct_ci(values: Any) -> str:
    try:
        return (
            f"{100.0 * float(values[0]):.2f}%–"
            f"{100.0 * float(values[1]):.2f}%"
        )
    except (TypeError, ValueError, IndexError):
        return "n/a"


def raw_ci(values: Any, digits: int = 5) -> str:
    try:
        return (
            f"{float(values[0]):.{digits}f}–"
            f"{float(values[1]):.{digits}f}"
        )
    except (TypeError, ValueError, IndexError):
        return "n/a"


def metric_rate(
    random_validation: dict[str, Any],
    key: str,
) -> float | None:
    value = (
        random_validation
        .get("metrics", {})
        .get(key, {})
        .get("rate")
    )

    return None if value is None else float(value)


def metric_count(
    random_validation: dict[str, Any],
    key: str,
) -> int | None:
    value = (
        random_validation
        .get("metrics", {})
        .get(key, {})
        .get("count")
    )

    return None if value is None else int(value)


def combine_dataset_summaries(
    by_dataset: dict[str, dict[str, Any]],
    dataset_keys: list[str],
) -> dict[str, Any]:
    missing = [key for key in dataset_keys if key not in by_dataset]

    if missing:
        raise KeyError(f"Missing H2 dataset summaries: {missing}")

    rows = [by_dataset[key] for key in dataset_keys]
    total_n = sum(int(row["n"]) for row in rows)

    if total_n <= 0:
        raise ValueError(f"Cannot combine empty H2 summaries: {dataset_keys}")

    combined: dict[str, Any] = {"n": total_n}
    weighted_fields = {
        "usable_rate",
        "critical_topology_error_rate",
        "skeleton_follows_ink_rate",
        "border_artifact_rate",
        "mean_graph_quality_0_3",
        "mean_cer",
        "mean_structural_risk_score",
    }

    for field in weighted_fields:
        values = [
            (int(row["n"]), row.get(field))
            for row in rows
            if row.get(field) is not None
        ]

        if values:
            combined[field] = sum(
                n * float(value) for n, value in values
            ) / sum(n for n, _ in values)

    count_fields = {
        "failure_stage_counts",
        "quality_counts",
        "critical_counts",
        "border_artifact_counts",
        "exclusion_reason_counts",
    }

    for field in count_fields:
        counts: dict[str, int] = {}

        for row in rows:
            for key, value in row.get(field, {}).items():
                counts[str(key)] = counts.get(str(key), 0) + int(value)

        if counts:
            combined[field] = counts

    return combined


def relative_status(row: dict[str, Any]) -> str:
    ci = (
        row.get("bootstrap", {})
        .get("relative_advantage_ci95", [None, None])
    )
    p_value = (
        row.get("permutation", {})
        .get("relative_advantage_one_sided_p")
    )

    try:
        if float(ci[0]) > 0 and float(p_value) < 0.05:
            return "supported"
    except (TypeError, ValueError, IndexError):
        pass

    estimate = row.get("relative_advantage")

    try:
        if float(estimate) <= 0:
            return "not supported"
    except (TypeError, ValueError):
        pass

    return "inconclusive"


def make_consolidated_report(
    evidence: dict[str, Any],
) -> str:
    h1 = evidence["h1"]
    h2 = evidence["h2"]
    h3 = evidence["h3"]
    recognition = evidence["recognition"]

    overall = h1["overall"]
    h1_desc = h1["descriptive"]

    image_desc = h1_desc["image_only"]
    graph_desc = h1_desc["graph_recomputed_v3"]

    h2_hc = h2["hkr_plus_cyrillic"]
    h2_old_school = h2["old_school_audit"]
    h2_random = h2["random_validation"]

    h3_best = h3["structural_core"]
    h3_corr = h3["best_global_correlation"]

    lines: list[str] = []

    lines.append("# HI-CSG-R consolidated evidence report — v2")
    lines.append("")

    lines.append("## 1. Executive verdict")
    lines.append("")
    lines.append("```text")
    lines.append("Overall result: mixed but scientifically informative")
    lines.append("Strong H1: rejected")
    lines.append("Partial relative-robustness H1: supported")
    lines.append("H2 visible-structure preservation: partially supported")
    lines.append("School Notebooks preprocessing repair: independently supported")
    lines.append("H3 graph diagnostics: localized partial support")
    lines.append("Absolute HTR improvement from graph fusion: not supported")
    lines.append("```")
    lines.append("")
    lines.append(
        "Canonical visible-stroke graph descriptors form a useful "
        "intermediate representation for robustness analysis, preprocessing "
        "validation, and high-error sample triage. They do not currently "
        "produce a recognizer that outperforms the image-only baseline in "
        "absolute character error rate."
    )
    lines.append("")

    lines.append("## 2. H1 — Robustness under visual distortions")
    lines.append("")
    lines.append("### 2.1 Primary paired result")
    lines.append("")
    lines.append("| metric | result |")
    lines.append("|---|---:|")
    lines.append(
        f"| image-only relative degradation | "
        f"{pct(overall['image_relative_degradation'])} |"
    )
    lines.append(
        f"| graph relative degradation | "
        f"{pct(overall['graph_relative_degradation'])} |"
    )
    lines.append(
        f"| relative robustness advantage | "
        f"{pct(overall['relative_advantage'])} |"
    )
    lines.append(
        f"| cluster-bootstrap 95% CI | "
        f"{pct_ci(overall['bootstrap']['relative_advantage_ci95'])} |"
    )
    lines.append(
        f"| paired permutation p | "
        f"{overall['permutation']['relative_advantage_one_sided_p']:.6f} |"
    )
    lines.append(
        f"| absolute degradation advantage | "
        f"{fmt(overall['absolute_advantage'], 5)} |"
    )
    lines.append(
        f"| absolute advantage 95% CI | "
        f"{raw_ci(overall['bootstrap']['absolute_advantage_ci95'])} |"
    )
    lines.append(
        f"| graph − image distorted CER | "
        f"{fmt(overall['distorted_cer_gap'], 5)} |"
    )
    lines.append("")
    lines.append(
        "The graph model has a statistically supported advantage in "
        "relative CER degradation. However, its absolute degradation is not "
        "better, and its distorted-image CER remains substantially higher."
    )
    lines.append("")

    lines.append("### 2.2 Descriptive absolute performance")
    lines.append("")
    lines.append(
        "| model | clean CER | mean distorted CER | "
        "mean absolute delta | relative degradation |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| `image_only` | "
        f"{fmt(image_desc['clean_cer'], 5)} | "
        f"{fmt(image_desc['mean_distorted_cer'], 5)} | "
        f"{fmt(image_desc['mean_absolute_delta'], 5)} | "
        f"{pct(image_desc['mean_relative_degradation'])} |"
    )
    lines.append(
        f"| `graph_vector_v2_recomputed` | "
        f"{fmt(graph_desc['clean_cer'], 5)} | "
        f"{fmt(graph_desc['mean_distorted_cer'], 5)} | "
        f"{fmt(graph_desc['mean_absolute_delta'], 5)} | "
        f"{pct(graph_desc['mean_relative_degradation'])} |"
    )
    lines.append("")

    lines.append("### 2.3 Distortion-family evidence")
    lines.append("")
    lines.append(
        "| family | image relative | graph relative | "
        "advantage | 95% CI | p | verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for key, row in sorted(h1["families"].items()):
        family = key.removeprefix("family:")

        lines.append(
            f"| `{family}` | "
            f"{pct(row['image_relative_degradation'])} | "
            f"{pct(row['graph_relative_degradation'])} | "
            f"{pct(row['relative_advantage'])} | "
            f"{pct_ci(row['bootstrap']['relative_advantage_ci95'])} | "
            f"{row['permutation']['relative_advantage_one_sided_p']:.6f} | "
            f"{relative_status(row)} |"
        )

    lines.append("")
    lines.append(
        "Relative robustness is supported for low contrast, additive noise, "
        "and stroke thinning. Blur is inconclusive under the combined "
        "bootstrap-and-permutation criterion, while stroke thickening shows "
        "no relative advantage."
    )
    lines.append("")

    lines.append("## 3. H2 — Visible-structure preservation")
    lines.append("")
    lines.append("### 3.1 Original diagnostic audit")
    lines.append("")
    lines.append(
        "| subset | n | critical topology error | "
        "skeleton follows ink | mean graph quality |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| `HKR + Cyrillic` | "
        f"{h2_hc['n']} | "
        f"{pct(h2_hc['critical_topology_error_rate'])} | "
        f"{pct(h2_hc['skeleton_follows_ink_rate'])} | "
        f"{fmt(h2_hc['mean_graph_quality_0_3'], 3)} |"
    )
    lines.append(
        f"| `School Notebooks, old preprocessing` | "
        f"{h2_old_school['n']} | "
        f"{pct(h2_old_school['critical_topology_error_rate'])} | "
        f"{pct(h2_old_school['skeleton_follows_ink_rate'])} | "
        f"{fmt(h2_old_school['mean_graph_quality_0_3'], 3)} |"
    )
    lines.append("")
    lines.append(
        "The original School Notebooks failure was localized to foreground "
        "extraction rather than to canonical graph construction itself."
    )
    lines.append("")

    lines.append("### 3.2 Independent random validation of foreground v3")
    lines.append("")
    lines.append("| metric | result |")
    lines.append("|---|---:|")
    lines.append(
        f"| random test samples | {h2_random['n']} |"
    )
    lines.append(
        f"| raw good-fix rate | "
        f"{pct(metric_rate(h2_random, 'good_fix'))} |"
    )
    lines.append(
        f"| partial-fix rate | "
        f"{pct(metric_rate(h2_random, 'partial_fix'))} |"
    )
    lines.append(
        f"| complete bad-fix rate | "
        f"{pct(metric_rate(h2_random, 'bad_fix'))} |"
    )
    lines.append(
        f"| strict usable rate | "
        f"{pct(metric_rate(h2_random, 'strict_usable'))} |"
    )
    lines.append(
        f"| real-ink loss rate | "
        f"{pct(metric_rate(h2_random, 'real_ink_erased'))} |"
    )
    lines.append(
        f"| residual background-artifact rate | "
        f"{pct(metric_rate(h2_random, 'background_artifact_after'))} |"
    )
    lines.append(
        f"| skeleton-follows-ink rate | "
        f"{pct(metric_rate(h2_random, 'skeleton_follows_ink_after'))} |"
    )
    lines.append("")
    lines.append(
        "`school_dark_auto` is therefore accepted as the School Notebooks "
        "foreground extraction method for subsequent graph processing. "
        "The result generalizes beyond the original diagnostic subset to "
        "the sampled test distribution."
    )
    lines.append("")

    lines.append("### 3.3 Controlled recognition cross-evaluation")
    lines.append("")
    lines.append(
        "| checkpoint | old features CER | v3 features CER | ΔCER |"
    )
    lines.append("|---|---:|---:|---:|")

    old_old = recognition["runs"]["old_model_old_features"]
    old_new = recognition["runs"]["old_model_new_features"]
    new_old = recognition["runs"]["new_model_old_features"]
    new_new = recognition["runs"]["new_model_new_features"]

    lines.append(
        f"| graph-v2 | "
        f"{fmt(old_old['cer'], 5)} | "
        f"{fmt(old_new['cer'], 5)} | "
        f"{signed(float(old_new['cer']) - float(old_old['cer']))} |"
    )
    lines.append(
        f"| graph-v3 retrain | "
        f"{fmt(new_old['cer'], 5)} | "
        f"{fmt(new_new['cer'], 5)} | "
        f"{signed(float(new_new['cer']) - float(new_old['cer']))} |"
    )
    lines.append("")
    lines.append(
        "Foreground v3 does not cause the graph-v3 checkpoint degradation. "
        "Both checkpoints improve slightly when evaluated with the repaired "
        "features, while the newly trained checkpoint remains worse under "
        "both manifests. The degradation is therefore attributed to the "
        "training run rather than to foreground repair."
    )
    lines.append("")

    lines.append("## 4. H3 — Graph-derived error diagnostics")
    lines.append("")
    lines.append("| metric | result |")
    lines.append("|---|---:|")
    lines.append(
        f"| best global correlation feature | "
        f"`{h3_corr.get('feature', 'n/a')}` |"
    )
    lines.append(
        f"| best global Spearman r | "
        f"{fmt(h3_corr.get('spearman_r'))} |"
    )
    lines.append(
        f"| best structural feature set | "
        f"`{h3_best.get('feature_set', 'n/a')}` |"
    )
    lines.append(
        f"| best subgroup | "
        f"`{h3_best.get('group', 'n/a')}` |"
    )
    lines.append(
        f"| subgroup n | {h3_best.get('n', 'n/a')} |"
    )
    lines.append(
        f"| ROC-AUC | {fmt(h3_best.get('roc_auc'))} |"
    )
    lines.append(
        f"| PR-AUC | {fmt(h3_best.get('pr_auc'))} |"
    )
    lines.append(
        f"| PR-AUC lift | "
        f"{fmt(h3_best.get('pr_auc_lift_over_base_rate'))} |"
    )
    lines.append(
        f"| top-20% precision | "
        f"{fmt(h3_best.get('top20_precision'))} |"
    )
    lines.append("")
    lines.append(
        "Global individual structural features have weak associations with "
        "recognition error. Multifeature descriptors provide useful but "
        "localized high-error detection, especially in the HKR word subset. "
        "They should be interpreted as sample-difficulty signals rather than "
        "direct graph-quality scores."
    )
    lines.append("")

    lines.append("## 5. Final hypothesis matrix")
    lines.append("")
    lines.append(
        "| hypothesis | final status | supported interpretation |"
    )
    lines.append("|---|---|---|")
    lines.append(
        "| H1 strong: graph-aware HTR is more robust overall | "
        "rejected | Absolute clean and distorted CER remain worse. |"
    )
    lines.append(
        "| H1 partial: graph model has lower relative sensitivity | "
        "supported | Paired corpus relative advantage is statistically supported. |"
    )
    lines.append(
        "| H2: visible structure is preserved | "
        "partially supported | Supported diagnostically for HKR/Cyrillic and "
        "after preprocessing repair for sampled School Notebooks data. |"
    )
    lines.append(
        "| H3: graph descriptors diagnose recognition difficulty | "
        "partially supported | Useful multifeature signal exists in localized strata. |"
    )
    lines.append(
        "| Graph fusion improves recognition accuracy | "
        "not supported | Current fusion models remain inferior to image-only HTR. |"
    )
    lines.append("")

    lines.append("## 6. Safe claims")
    lines.append("")
    for claim in evidence["safe_claims"]:
        lines.append(f"- {claim}")
    lines.append("")

    lines.append("## 7. Claims to avoid")
    lines.append("")
    for claim in evidence["unsafe_claims"]:
        lines.append(f"- {claim}")
    lines.append("")

    lines.append("## 8. Final project framing")
    lines.append("")
    lines.append(
        "The contribution is not a new top-performing recognizer. "
        "It is a controlled study of canonical visible-stroke graph "
        "descriptors as an intermediate structural representation for "
        "offline handwriting. The representation provides measurable value "
        "for relative robustness analysis, preprocessing validation, and "
        "localized failure triage, while the experiments also identify the "
        "limits of direct global graph-vector fusion."
    )

    return "\n".join(lines)


def make_results(evidence: dict[str, Any]) -> str:
    h1 = evidence["h1"]
    h2 = evidence["h2"]
    h3 = evidence["h3"]
    recognition = evidence["recognition"]

    overall = h1["overall"]
    random_validation = h2["random_validation"]
    h2_hc = h2["hkr_plus_cyrillic"]
    h3_best = h3["structural_core"]

    old_old = recognition["runs"]["old_model_old_features"]
    old_new = recognition["runs"]["old_model_new_features"]
    new_new = recognition["runs"]["new_model_new_features"]

    lines: list[str] = []

    lines.append("# Results")
    lines.append("")

    lines.append("## Recognition baselines and graph fusion")
    lines.append("")
    lines.append(
        "The image-only recognizer remained the strongest model in absolute "
        "recognition quality. The retained graph-vector checkpoint achieved "
        f"a clean CER of {float(old_old['cer']):.4f}, while replacing its "
        f"School Notebooks graph features with the repaired foreground-v3 "
        f"features changed CER only slightly to {float(old_new['cer']):.4f}. "
        f"A newly trained graph-fusion checkpoint reached a worse CER of "
        f"{float(new_new['cer']):.4f}. Cross-evaluation showed that this "
        "degradation persisted with both old and repaired manifests, so it "
        "was not caused by foreground v3."
    )
    lines.append("")

    lines.append("## H1: robustness")
    lines.append("")
    lines.append(
        "Across the 5,563 clean test samples and 15 distortion conditions, "
        "the paired corpus-level analysis yielded an image-only relative CER "
        f"degradation of {pct(overall['image_relative_degradation'])} and a "
        f"graph-model degradation of {pct(overall['graph_relative_degradation'])}. "
        f"The resulting relative robustness advantage was "
        f"{pct(overall['relative_advantage'])}, with a 95% paired "
        f"cluster-bootstrap interval of "
        f"{pct_ci(overall['bootstrap']['relative_advantage_ci95'])} and a "
        f"one-sided permutation p-value of "
        f"{overall['permutation']['relative_advantage_one_sided_p']:.6f}."
    )
    lines.append("")
    lines.append(
        f"The absolute degradation advantage was "
        f"{float(overall['absolute_advantage']):.5f}, with a 95% interval of "
        f"{raw_ci(overall['bootstrap']['absolute_advantage_ci95'])}. "
        f"Furthermore, distorted CER remained higher for the graph model by "
        f"{float(overall['distorted_cer_gap']):.5f}. Thus, the graph model "
        "was relatively less sensitive to distortion but remained the worse "
        "recognizer in absolute terms."
    )
    lines.append("")

    lines.append("## H2: visible-structure preservation")
    lines.append("")
    lines.append(
        f"In the original diagnostic audit, the combined HKR and Cyrillic "
        f"subset contained {h2_hc['n']} samples. The critical-topology-error "
        f"rate was {pct(h2_hc['critical_topology_error_rate'])}, while the "
        f"skeleton followed visible ink in "
        f"{pct(h2_hc['skeleton_follows_ink_rate'])} of samples. "
        "The initial School Notebooks failure was traced to foreground "
        "extraction rather than to graph construction."
    )
    lines.append("")
    lines.append(
        f"An independent random validation on {random_validation['n']} School "
        f"Notebooks test samples selected `school_dark_auto` in all inspected "
        f"cases. The raw good-fix rate was "
        f"{pct(metric_rate(random_validation, 'good_fix'))}, while the strict "
        f"usable rate was "
        f"{pct(metric_rate(random_validation, 'strict_usable'))}. "
        f"Real-ink loss was observed in "
        f"{pct(metric_rate(random_validation, 'real_ink_erased'))} of samples, "
        f"residual background artifacts in "
        f"{pct(metric_rate(random_validation, 'background_artifact_after'))}, "
        f"and the resulting skeleton followed visible ink in "
        f"{pct(metric_rate(random_validation, 'skeleton_follows_ink_after'))}."
    )
    lines.append("")

    lines.append("## H3: diagnostic value")
    lines.append("")
    lines.append(
        f"The strongest multifeature diagnostic result used "
        f"`{h3_best.get('feature_set', 'structural_core')}` in the subgroup "
        f"`{h3_best.get('group', 'n/a')}` with "
        f"n={h3_best.get('n', 'n/a')}. It achieved ROC-AUC "
        f"{fmt(h3_best.get('roc_auc'))}, PR-AUC "
        f"{fmt(h3_best.get('pr_auc'))}, and top-20% precision "
        f"{fmt(h3_best.get('top20_precision'))}. "
        "The signal was therefore useful but localized rather than global."
    )

    return "\n".join(lines)


def make_discussion(evidence: dict[str, Any]) -> str:
    h1 = evidence["h1"]
    h2 = evidence["h2"]
    h3 = evidence["h3"]

    overall = h1["overall"]
    random_validation = h2["random_validation"]
    h3_best = h3["structural_core"]

    lines: list[str] = []

    lines.append("# Discussion")
    lines.append("")

    lines.append("## Relative robustness without absolute superiority")
    lines.append("")
    lines.append(
        "The main robustness result is deliberately narrower than a claim of "
        "superior recognition. The graph-vector model exhibited a "
        f"{pct(overall['relative_advantage'])} reduction in relative CER "
        "degradation compared with the image-only baseline, and this effect "
        "was supported by paired cluster bootstrap and permutation testing. "
        "At the same time, the graph model started from a substantially worse "
        "clean CER and remained worse on distorted images. Relative stability "
        "therefore indicates lower sensitivity around the model's own error "
        "level, not a better HTR system."
    )
    lines.append("")
    lines.append(
        "This distinction is important because a weaker model can show a "
        "smaller proportional degradation partly because its initial error "
        "rate is already high. The paired analysis reduces, but does not "
        "eliminate, this interpretive limitation. For that reason, strong H1 "
        "is rejected and only a partial sensitivity claim is retained."
    )
    lines.append("")

    lines.append("## Meaning of foreground repair")
    lines.append("")
    lines.append(
        "The School Notebooks investigation demonstrates that graph quality "
        "depends critically on upstream foreground extraction. The original "
        "skeleton failures were not evidence that the visible-stroke graph "
        "abstraction was intrinsically unsuitable. They were caused by page "
        "background being classified as foreground before skeletonization."
    )
    lines.append("")
    lines.append(
        f"The independent random validation produced a strict usable rate of "
        f"{pct(metric_rate(random_validation, 'strict_usable'))}. This supports "
        "the generality of `school_dark_auto` within the sampled School "
        "Notebooks test distribution. Nevertheless, the remaining ink-loss "
        "and residual-artifact cases show that the preprocessing rule is not "
        "perfect and should not be treated as universal."
    )
    lines.append("")

    lines.append("## Why graph repair did not improve recognition")
    lines.append("")
    lines.append(
        "Improving visible graph quality did not materially improve the "
        "graph-fusion recognizer. This is not contradictory. A graph can be "
        "more faithful as a structural description while still adding little "
        "information beyond the convolutional image representation, or while "
        "being fused at an ineffective architectural location. The controlled "
        "cross-evaluation indicates that foreground v3 was compatible with "
        "the retained graph-v2 checkpoint, but a new training run was worse "
        "independently of which feature manifest was used."
    )
    lines.append("")
    lines.append(
        "The result therefore separates representation quality from fusion "
        "utility: a cleaner structural representation is useful for analysis "
        "and diagnostics, but it is not sufficient by itself to improve CTC "
        "recognition."
    )
    lines.append("")

    lines.append("## Diagnostic role of graph descriptors")
    lines.append("")
    lines.append(
        f"The strongest H3 result reached ROC-AUC "
        f"{fmt(h3_best.get('roc_auc'))} in "
        f"`{h3_best.get('group', 'n/a')}`. This level is meaningful for "
        "ranking or triage but insufficient for a general error predictor. "
        "The weak global correlations and localized multifeature gains suggest "
        "that structural difficulty interacts with dataset, text level, and "
        "writing style."
    )
    lines.append("")
    lines.append(
        "Graph-derived risk should therefore be used to prioritize manual "
        "inspection, detect suspicious preprocessing, or stratify evaluation. "
        "It should not be interpreted as a calibrated measurement of graph "
        "correctness."
    )
    lines.append("")

    lines.append("## Main contribution")
    lines.append("")
    lines.append(
        "The strongest contribution of the project is a reproducible "
        "visible-stroke structural layer between offline handwriting images "
        "and recognition output. Its value lies in making preprocessing and "
        "structural failure modes measurable. The negative recognition result "
        "is also informative: simple global graph-vector fusion is not enough "
        "to convert structural descriptors into improved transcription."
    )

    return "\n".join(lines)


def make_limitations(evidence: dict[str, Any]) -> str:
    h2_random = evidence["h2"]["random_validation"]

    lines: list[str] = []

    lines.append("# Limitations")
    lines.append("")

    lines.append("## Absolute recognition performance")
    lines.append("")
    lines.append(
        "The graph-aware models do not outperform the image-only baseline in "
        "clean or distorted absolute CER. All robustness claims must therefore "
        "be stated as relative sensitivity results rather than recognition "
        "superiority."
    )
    lines.append("")

    lines.append("## Synthetic distortion protocol")
    lines.append("")
    lines.append(
        "The robustness evaluation uses synthetic blur, additive noise, "
        "contrast reduction, and morphological stroke changes. These "
        "perturbations are controlled and reproducible but do not cover the "
        "full distribution of real scanning, camera, compression, paper, ink, "
        "and page-layout degradation."
    )
    lines.append("")

    lines.append("## Relative-degradation estimand")
    lines.append("")
    lines.append(
        "Relative degradation depends on each model's clean error rate. "
        "Because the graph model starts from a worse baseline, proportional "
        "changes can favor it even when absolute errors remain higher. The "
        "study reports corpus-level relative and absolute effects separately "
        "to prevent these quantities from being conflated."
    )
    lines.append("")

    lines.append("## Manual graph audit")
    lines.append("")
    lines.append(
        "The initial H2 audit was diagnostically selected across error and "
        "risk strata and should not be interpreted as a population estimate. "
        "Its purpose was failure-mode discovery and structural inspection."
    )
    lines.append("")

    lines.append("## Random School Notebooks validation")
    lines.append("")
    lines.append(
        f"The independent validation includes {h2_random['n']} randomly "
        "sampled items from one test split. It does not establish performance "
        "on all splits, all notebook sources, or unseen acquisition settings. "
        "The annotations were produced by one evaluator, and no inter-rater "
        "agreement estimate is available."
    )
    lines.append("")

    lines.append("## Remaining preprocessing errors")
    lines.append("")
    lines.append(
        f"Foreground v3 retained background artifacts in "
        f"{pct(metric_rate(h2_random, 'background_artifact_after'))} of the "
        f"random validation sample and removed some visible ink in "
        f"{pct(metric_rate(h2_random, 'real_ink_erased'))}. The method is "
        "therefore a substantial repair, not a perfect segmentation solution."
    )
    lines.append("")

    lines.append("## Lack of gold graph topology")
    lines.append("")
    lines.append(
        "The project does not contain exhaustive node-edge gold annotations "
        "for the visible-stroke graph. Most automated graph-quality variables "
        "are structural proxies. Manual skeleton inspection establishes "
        "plausibility but not exact topological accuracy."
    )
    lines.append("")

    lines.append("## Model and optimization variance")
    lines.append("")
    lines.append(
        "The controlled graph-fusion retraining experiment produced a worse "
        "checkpoint under both old and repaired feature manifests. This "
        "demonstrates sensitivity to training dynamics, but a single retrain "
        "does not quantify full seed-to-seed variance."
    )
    lines.append("")

    lines.append("## Localized H3 evidence")
    lines.append("")
    lines.append(
        "The strongest graph-based high-error detection result is localized "
        "to a particular dataset and text-level subgroup. It should not be "
        "generalized to all samples or interpreted as a universal confidence "
        "estimator."
    )
    lines.append("")

    lines.append("## Offline visible structure, not pen trajectory")
    lines.append("")
    lines.append(
        "The generated graph represents reproducible visible stroke structure "
        "in a static image. It does not reconstruct writing order, pen lifts, "
        "pressure, velocity, or the true online trajectory."
    )

    return "\n".join(lines)


def make_claim_matrix(evidence: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Final claim matrix — v2")
    lines.append("")
    lines.append("| claim | status | required wording |")
    lines.append("|---|---|---|")
    lines.append(
        "| Graph-aware HTR outperforms image-only HTR. | "
        "not supported | Do not claim. |"
    )
    lines.append(
        "| Graph-vector HTR is less sensitive to tested distortions in "
        "relative CER terms. | supported | State together with worse absolute CER. |"
    )
    lines.append(
        "| Graph-vector HTR has lower absolute degradation. | "
        "not supported | Report the negative absolute-advantage estimate. |"
    )
    lines.append(
        "| The graph pipeline preserves visible structure on audited "
        "HKR/Cyrillic samples. | partially supported | "
        "Restrict to the diagnostic audit evidence. |"
    )
    lines.append(
        "| `school_dark_auto` repairs School Notebooks foreground extraction. | "
        "supported on sampled test distribution | Report random-100 rates and "
        "remaining ink-loss/artifact failures. |"
    )
    lines.append(
        "| Foreground v3 improves graph-fusion recognition. | "
        "not supported | It gives only a very small inference-time CER change. |"
    )
    lines.append(
        "| Graph descriptors identify difficult samples. | "
        "partially supported | Restrict to localized multifeature results. |"
    )
    lines.append(
        "| Structural risk directly measures graph correctness. | "
        "not supported | Describe it as a hard-sample indicator. |"
    )
    lines.append(
        "| The graph reconstructs real pen trajectory. | "
        "not supported by design | Use visible-stroke structural representation. |"
    )
    lines.append("")

    lines.append("## Frozen thesis claim")
    lines.append("")
    lines.append(
        "> Canonical visible-stroke graph descriptors provide a reproducible "
        "intermediate representation for offline handwriting analysis. They "
        "show statistically supported value for relative robustness analysis, "
        "foreground-preprocessing validation, and localized recognition-error "
        "triage. However, current graph-fusion models do not outperform a "
        "strong image-only recognizer in absolute character error rate."
    )

    return "\n".join(lines)


def make_status() -> str:
    lines: list[str] = []

    lines.append("# Final experiment status — v2")
    lines.append("")

    lines.append("## Completed")
    lines.append("")
    lines.append("- Image-only baseline evaluation.")
    lines.append("- Graph-vector and gated-fusion evaluation.")
    lines.append("- Synthetic robustness evaluation across 15 conditions.")
    lines.append("- End-to-end recomputation of graph features under distortion.")
    lines.append("- Paired cluster bootstrap and permutation analysis for H1.")
    lines.append("- H2 manual diagnostic audit.")
    lines.append("- School Notebooks foreground failure diagnosis.")
    lines.append("- Deterministic `school_dark_auto` preprocessing repair.")
    lines.append("- Independent random-100 foreground validation.")
    lines.append("- Graph-feature cross-evaluation with old and new checkpoints.")
    lines.append("- H3 graph-derived high-error analysis.")
    lines.append("- Consolidated evidence and manuscript text generation.")
    lines.append("")

    lines.append("## Experimental freeze")
    lines.append("")
    lines.append("- Do not add new HTR architectures.")
    lines.append("- Do not perform further graph-fusion CER tuning.")
    lines.append("- Do not retrain graph-v3 again.")
    lines.append("- Do not add new synthetic corruption families for this study.")
    lines.append("- Do not reinterpret relative robustness as absolute superiority.")
    lines.append("")

    lines.append("## Remaining work")
    lines.append("")
    lines.append("- Integrate Results, Discussion, and Limitations into the manuscript.")
    lines.append("- Prepare final figures from existing experiment outputs.")
    lines.append("- Write Methods with exact data splits and preprocessing definitions.")
    lines.append("- Verify all manuscript numbers against generated JSON evidence.")
    lines.append("- Freeze tables and archive reproducibility commands.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--h1_json", required=True)
    parser.add_argument("--h2_manual_json", required=True)
    parser.add_argument("--h2_random_json", required=True)
    parser.add_argument("--school_controlled_json", required=True)
    parser.add_argument("--cross_eval_json", required=True)
    parser.add_argument("--h3_json", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    h1 = load_json(args.h1_json)
    h2_manual = load_json(args.h2_manual_json)
    h2_random = load_json(args.h2_random_json)
    school_controlled = load_json(args.school_controlled_json)
    cross_eval = load_json(args.cross_eval_json)
    h3 = load_json(args.h3_json)

    h1_descriptive = h1.get(
        "descriptive_condition_average",
        {},
    )

    required_h1_modes = {
        "image_only",
        "graph_recomputed_v3",
    }
    missing_h1_modes = required_h1_modes - set(h1_descriptive)

    if missing_h1_modes:
        raise KeyError(
            f"Missing H1 descriptive modes: "
            f"{sorted(missing_h1_modes)}"
        )

    h3_best = (
        h3.get("best_by_feature_set", {})
        .get("structural_core")
    )

    if not h3_best:
        raise KeyError(
            "No structural_core result in H3 summary"
        )

    h2_by_dataset = h2_manual.get("by_dataset", {})
    h2_hkr_plus_cyrillic = h2_manual.get(
        "hkr_plus_cyrillic"
    ) or combine_dataset_summaries(
        h2_by_dataset,
        ["hkr_words", "cyrillic_handwriting"],
    )

    evidence = {
        "version": "hi_csg_r_consolidated_evidence_v2",
        "sources": {
            "h1_json": args.h1_json,
            "h2_manual_json": args.h2_manual_json,
            "h2_random_json": args.h2_random_json,
            "school_controlled_json": args.school_controlled_json,
            "cross_eval_json": args.cross_eval_json,
            "h3_json": args.h3_json,
        },
        "overall_verdict": "mixed_partial_support",
        "h1": {
            "verdict": h1["verdict"],
            "overall": h1["overall"],
            "datasets": h1.get("datasets", {}),
            "families": h1.get("families", {}),
            "descriptive": h1_descriptive,
        },
        "h2": {
            "verdict": {
                "visible_structure": "partial_support",
                "school_foreground_repair": (
                    "independently_supported_on_random_test_sample"
                ),
                "uniform_population_level_topology": "not_established",
            },
            "hkr_plus_cyrillic": h2_hkr_plus_cyrillic,
            "by_dataset_original": h2_manual.get(
                "by_dataset",
                {},
            ),
            "old_school_audit": (
                h2_manual["by_dataset"][
                    "school_notebooks_clean"
                ]
            ),
            "random_validation": h2_random,
            "controlled_conclusion": school_controlled,
        },
        "h3": {
            "verdict": "localized_partial_support",
            "structural_core": h3_best,
            "best_global_correlation": (
                h3.get("best_abs_spearman") or {}
            ),
            "dataset_summary": h3.get(
                "dataset_summary",
                {},
            ),
        },
        "recognition": cross_eval,
        "safe_claims": [
            (
                "The graph-vector model has a statistically supported "
                "relative robustness advantage under the tested distortions."
            ),
            (
                "The graph-vector model remains worse than the image-only "
                "baseline in clean and distorted absolute CER."
            ),
            (
                "`school_dark_auto` substantially repairs School Notebooks "
                "foreground extraction on an independently sampled test subset."
            ),
            (
                "Foreground repair improves visible structural extraction but "
                "does not materially improve graph-fusion recognition."
            ),
            (
                "Multifeature graph descriptors provide localized value for "
                "high-error sample triage."
            ),
            (
                "The generated graph describes visible static stroke structure, "
                "not the true writing trajectory."
            ),
        ],
        "unsafe_claims": [
            "Do not claim graph-aware recognition is superior to image-only recognition.",
            "Do not claim that strong H1 is confirmed.",
            (
                "Do not claim uniform topology preservation across all "
                "datasets and acquisition settings."
            ),
            (
                "Do not treat the random-100 School Notebooks result as "
                "evidence for all handwriting domains."
            ),
            (
                "Do not claim that foreground v3 improves recognition "
                "accuracy in a practically meaningful way."
            ),
            (
                "Do not describe structural risk as direct graph-quality "
                "ground truth."
            ),
            "Do not describe the graph as reconstructed pen trajectory.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        out_dir / "hi_csg_r_consolidated_evidence_report_v2.json",
        evidence,
    )
    write_text(
        out_dir / "hi_csg_r_consolidated_evidence_report_v2.md",
        make_consolidated_report(evidence),
    )
    write_text(
        out_dir / "manuscript_results_v2.md",
        make_results(evidence),
    )
    write_text(
        out_dir / "manuscript_discussion_v2.md",
        make_discussion(evidence),
    )
    write_text(
        out_dir / "manuscript_limitations_v2.md",
        make_limitations(evidence),
    )
    write_text(
        out_dir / "final_claim_matrix_v2.md",
        make_claim_matrix(evidence),
    )
    write_text(
        out_dir / "final_experiment_status_v2.md",
        make_status(),
    )


if __name__ == "__main__":
    main()
