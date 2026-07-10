from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
from pathlib import Path
from typing import Any


PACKAGE_VERSION = "hi_csg_r_final_documentation_v3"


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")

    return value


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


def pct_ci(value: Any) -> str:
    try:
        return (
            f"{100.0 * float(value[0]):.2f}%–"
            f"{100.0 * float(value[1]):.2f}%"
        )
    except (TypeError, ValueError, IndexError):
        return "n/a"


def raw_ci(value: Any, digits: int = 5) -> str:
    try:
        return (
            f"{float(value[0]):.{digits}f}–"
            f"{float(value[1]):.{digits}f}"
        )
    except (TypeError, ValueError, IndexError):
        return "n/a"


def metric(
    random_validation: dict[str, Any],
    key: str,
    field: str = "rate",
) -> Any:
    return (
        random_validation
        .get("metrics", {})
        .get(key, {})
        .get(field)
    )


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

    try:
        if float(row.get("relative_advantage")) <= 0:
            return "not supported"
    except (TypeError, ValueError):
        pass

    return "inconclusive"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Evidence normalization
# ---------------------------------------------------------------------

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


def build_evidence(
    *,
    h1: dict[str, Any],
    h2_manual: dict[str, Any],
    h2_random: dict[str, Any],
    school_controlled: dict[str, Any],
    cross_eval: dict[str, Any],
    h3: dict[str, Any],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    h1_overall = h1.get("overall")

    if not h1_overall:
        raise KeyError("H1 JSON has no overall result")

    h1_descriptive = h1.get(
        "descriptive_condition_average",
        {},
    )

    for mode in [
        "image_only",
        "graph_frozen_clean",
        "graph_recomputed_v3",
    ]:
        if mode not in h1_descriptive:
            raise KeyError(
                f"H1 JSON has no descriptive mode {mode!r}"
            )

    h2_hc = h2_manual.get("hkr_plus_cyrillic")
    h2_by_dataset = h2_manual.get("by_dataset", {})

    if not h2_hc:
        h2_hc = combine_dataset_summaries(
            h2_by_dataset,
            ["hkr_words", "cyrillic_handwriting"],
        )

    if "school_notebooks_clean" not in h2_by_dataset:
        raise KeyError(
            "H2 JSON has no school_notebooks_clean"
        )

    h3_structural = (
        h3.get("best_by_feature_set", {})
        .get("structural_core")
    )

    if not h3_structural:
        raise KeyError(
            "H3 JSON has no structural_core result"
        )

    runs = cross_eval.get("runs", {})

    required_runs = {
        "old_model_old_features",
        "old_model_new_features",
        "new_model_old_features",
        "new_model_new_features",
    }

    missing_runs = required_runs - set(runs)

    if missing_runs:
        raise KeyError(
            f"Missing cross-evaluation runs: "
            f"{sorted(missing_runs)}"
        )

    evidence = {
        "version": PACKAGE_VERSION,
        "source_paths": source_paths,
        "overall_verdict": {
            "strong_h1_supported": False,
            "partial_relative_h1_supported": True,
            "absolute_htr_improvement_supported": False,
            "h2_visible_structure": "partial_support",
            "school_foreground_repair": (
                "supported_on_independent_random_test_sample"
            ),
            "h3_diagnostic_utility": (
                "localized_partial_support"
            ),
            "graph_fusion_improves_recognition": False,
        },
        "h1": {
            "overall": h1_overall,
            "datasets": h1.get("datasets", {}),
            "families": h1.get("families", {}),
            "descriptive": h1_descriptive,
        },
        "h2": {
            "hkr_plus_cyrillic": h2_hc,
            "original_by_dataset": h2_by_dataset,
            "old_school_audit": (
                h2_by_dataset["school_notebooks_clean"]
            ),
            "random_validation": h2_random,
            "controlled_conclusion": school_controlled,
        },
        "h3": {
            "structural_core": h3_structural,
            "best_global_correlation": (
                h3.get("best_abs_spearman") or {}
            ),
            "dataset_summary": (
                h3.get("dataset_summary") or {}
            ),
        },
        "recognition_cross_evaluation": cross_eval,
        "frozen_claim": (
            "Canonical visible-stroke graph descriptors provide a "
            "reproducible intermediate representation for offline "
            "handwriting analysis. They show statistically supported "
            "value for relative robustness analysis, foreground "
            "preprocessing validation, and localized recognition-error "
            "triage. Current graph-fusion models do not outperform a "
            "strong image-only recognizer in absolute character error rate."
        ),
        "safe_claims": [
            (
                "The graph-vector model has a statistically supported "
                "relative robustness advantage under the tested "
                "synthetic distortions."
            ),
            (
                "The graph-vector model remains worse than the "
                "image-only baseline in clean and distorted absolute CER."
            ),
            (
                "`school_dark_auto` substantially repairs School "
                "Notebooks foreground extraction on an independently "
                "sampled test subset."
            ),
            (
                "Improved visible graph extraction does not materially "
                "improve the tested graph-fusion recognizer."
            ),
            (
                "Multifeature structural descriptors provide localized "
                "value for high-error sample triage."
            ),
            (
                "The generated graph describes visible static stroke "
                "structure and not the true online pen trajectory."
            ),
        ],
        "unsafe_claims": [
            "Graph-aware recognition is superior to image-only recognition.",
            "Strong H1 is confirmed.",
            (
                "Visible graph topology is preserved uniformly across "
                "all datasets and acquisition conditions."
            ),
            (
                "The random-100 School Notebooks result generalizes to "
                "all handwriting domains."
            ),
            (
                "Foreground v3 materially improves recognition accuracy."
            ),
            (
                "Structural risk is a direct gold measurement of graph "
                "correctness."
            ),
            "The graph recovers the true online writing trajectory.",
        ],
    }

    return evidence


# ---------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------

def make_index() -> str:
    return """
# HI-CSG-R final documentation package — v3

This directory is the frozen descriptive and evidential documentation
package for the HI-CSG-R experiments.

## Reading order

1. `01_EXECUTIVE_SUMMARY.md`
2. `02_METHODS.md`
3. `03_RESULTS.md`
4. `04_DISCUSSION.md`
5. `05_LIMITATIONS.md`
6. `06_CONCLUSION.md`
7. `07_REPRODUCIBILITY.md`
8. `08_FINAL_TABLES.md`
9. `09_FIGURE_PLAN.md`
10. `10_CLAIM_MATRIX.md`
11. `11_EXPERIMENT_REGISTRY.md`
12. `ABSTRACT_RU.md`
13. `ABSTRACT_EN.md`
14. `MANUSCRIPT_FULL.md`

## Machine-readable files

- `evidence_manifest.json`
- `documentation_validation.json`
- `SHA256SUMS`

## Scope

The package documents:

- canonical visible-stroke graph extraction;
- image-only and graph-aware HTR evaluation;
- robustness testing under 15 synthetic distortion conditions;
- paired cluster-bootstrap and permutation statistics;
- manual graph audit;
- School Notebooks foreground repair;
- independent random-100 validation;
- graph-feature cross-evaluation;
- graph-derived high-error diagnostics.

No additional HTR architecture experiments are required for the present
study.
"""


def make_executive_summary(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    random_v = e["h2"]["random_validation"]
    h3 = e["h3"]["structural_core"]

    return f"""
# Executive summary

## Research objective

The project investigates canonical graph descriptors of visible
handwriting strokes as an intermediate representation for offline
Russian-English handwritten text recognition.

The representation is intentionally not an estimate of pen order,
velocity, pressure, or real online trajectory. It describes reproducible
visible structure extracted from a static image.

## Main findings

### Relative robustness

The graph-vector recognizer shows lower relative CER degradation than
the image-only recognizer under the tested visual distortions.

- image-only relative degradation:
  **{pct(h1["image_relative_degradation"])}**
- graph-model relative degradation:
  **{pct(h1["graph_relative_degradation"])}**
- relative advantage:
  **{pct(h1["relative_advantage"])}**
- 95% paired cluster-bootstrap interval:
  **{pct_ci(h1["bootstrap"]["relative_advantage_ci95"])}**
- one-sided paired permutation p:
  **{h1["permutation"]["relative_advantage_one_sided_p"]:.6f}**

This is a relative sensitivity result, not an absolute recognition
advantage.

### Absolute recognition

The graph model remains worse on distorted images by
**{fmt(h1["distorted_cer_gap"], 5)} CER**.

Strong H1 is therefore rejected.

### Visible graph quality

The original School Notebooks graph failures were traced to foreground
extraction. The deterministic `school_dark_auto` method was validated on
an independent random sample of {random_v["n"]} test items.

- raw good-fix rate:
  **{pct(metric(random_v, "good_fix"))}**
- strict usable rate:
  **{pct(metric(random_v, "strict_usable"))}**
- skeleton-follows-ink rate:
  **{pct(metric(random_v, "skeleton_follows_ink_after"))}**
- real-ink loss:
  **{pct(metric(random_v, "real_ink_erased"))}**
- residual background artifacts:
  **{pct(metric(random_v, "background_artifact_after"))}**

### Diagnostic value

The strongest structural high-error detector was localized to
`{h3.get("group", "n/a")}`.

- ROC-AUC: **{fmt(h3.get("roc_auc"))}**
- PR-AUC: **{fmt(h3.get("pr_auc"))}**
- top-20% precision: **{fmt(h3.get("top20_precision"))}**

## Final interpretation

The graph representation is useful for:

- relative robustness analysis;
- preprocessing validation;
- structural inspection;
- localized failure triage.

It is not currently a successful replacement for the image-only
recognizer and does not provide superior absolute CER.
"""


def make_methods(e: dict[str, Any]) -> str:
    return """
# Methods

## 1. Problem formulation

Let an offline grayscale handwriting crop be denoted by \(I\). The
pipeline constructs a deterministic foreground mask \(F\), a skeleton
\(S\), and a set of graph-derived structural descriptors \(g(I)\).

The target representation is a canonical graph of visible stroke
structure. It is not intended to reconstruct the actual temporal pen
trajectory.

The recognition task predicts a character sequence from the image,
optionally conditioned on the graph descriptor vector.

## 2. Data organization

The final mixed test evaluation contains 5,563 samples:

- 1,563 Cyrillic Handwriting samples;
- 2,000 HKR samples;
- 2,000 School Notebooks samples.

Dataset identity is retained in all manifests for grouped evaluation.

Train, validation, and test partitions are represented by JSONL
manifests. Each graph-ready row contains:

- `sample_id`;
- image path;
- target transcription;
- dataset and text-level metadata;
- `graph_feature_names`;
- `graph_features`;
- graph warning information;
- preprocessing metadata.

## 3. Image preprocessing

Images are converted to grayscale.

For Cyrillic Handwriting and HKR, foreground extraction uses the
dataset-specific thresholding configuration established in the graph
feature extractor.

For School Notebooks, the final method is `school_dark_auto`:

1. threshold dark pixels at intensity 145;
2. remove connected foreground objects smaller than 4 pixels;
3. calculate foreground fraction;
4. retain threshold 145 if foreground fraction is at most 0.35;
5. otherwise repeat extraction with threshold 120.

The method was introduced because local adaptive binarization frequently
classified darker notebook background as foreground.

## 4. Skeleton and graph descriptors

The binary foreground mask is skeletonized.

The final graph vector contains 39 non-textual descriptors covering:

- crop width, height, and aspect ratio;
- foreground fraction;
- foreground bounding-box geometry;
- connected-component statistics;
- skeleton pixel fraction and component count;
- graph node and edge counts;
- average degree;
- endpoint, branch-point, and isolated-node counts;
- degree histogram;
- horizontal, vertical, and diagonal direction fractions;
- stroke-width statistics;
- graph warning count.

`text_len` is excluded from recognition and diagnostic feature sets to
prevent target-length leakage.

## 5. Recognition models

### Image-only model

The primary baseline is a convolutional recurrent CTC recognizer using
only the grayscale image.

### Graph-vector fusion model

The graph-vector model contains:

- a convolutional image encoder;
- adaptive vertical pooling;
- a graph MLP;
- temporal broadcasting of the global graph embedding;
- concatenation of image and graph representations;
- a bidirectional recurrent sequence encoder;
- a CTC classifier.

Graph features are standardized using mean and standard deviation
estimated from the training manifest.

### Controlled foreground cross-evaluation

To separate preprocessing effects from training-run effects, two
checkpoints were evaluated with two feature manifests:

- old checkpoint + old graph features;
- old checkpoint + foreground-v3 graph features;
- new checkpoint + old graph features;
- new checkpoint + foreground-v3 graph features.

## 6. Robustness protocol

Five distortion families were evaluated at three severity levels:

| family | mild | medium | strong |
|---|---:|---:|---:|
| Gaussian blur kernel | 3 | 5 | 7 |
| Gaussian-noise sigma | 8 | 16 | 24 |
| contrast alpha | 0.75 | 0.55 | 0.40 |
| stroke thinning kernel | 2 | 2 | 3 |
| stroke thickening kernel | 2 | 2 | 3 |

This produces 15 distorted conditions per source sample.

Graph features were recomputed from each distorted image. This prevents
the graph branch from receiving inherited clean-image descriptors.

## 7. Robustness estimands

For each model:

\[
D_{abs} = CER_{distorted} - CER_{clean}
\]

\[
D_{rel} =
\\frac{CER_{distorted} - CER_{clean}}
     {CER_{clean}}
\]

The primary relative robustness advantage is:

\[
A_{rel} =
D_{rel}^{image}
-
D_{rel}^{graph}
\]

Positive \(A_{rel}\) means that the graph model degrades less in
relative terms.

Absolute distorted CER is reported separately.

## 8. Statistical analysis

The inferential robustness analysis uses:

- all 5,563 clean source samples;
- all 15 distortion conditions;
- cluster resampling by clean source sample;
- 5,000 paired bootstrap iterations;
- 20,000 paired permutations.

All distortion observations belonging to a clean sample remain in the
same resampling cluster.

The primary confidence interval concerns corpus-level relative
degradation advantage.

## 9. H2 manual audit

A diagnostic audit examined graph extraction quality across:

- HKR;
- Cyrillic Handwriting;
- School Notebooks.

The audit recorded:

- usability;
- critical topology errors;
- whether the skeleton follows visible ink;
- border/background artifacts;
- graph quality on a 0–3 scale;
- inferred failure stage.

The initial audit subset was selected diagnostically and is not used as
a population estimate.

## 10. Independent School Notebooks validation

After development of `school_dark_auto`, an independent random sample of
100 School Notebooks test items was evaluated.

The strict usable criterion required:

- no visible real-ink removal;
- no remaining dominant background artifact;
- skeleton following the visible handwriting.

Wilson intervals were calculated for proportion estimates.

## 11. H3 diagnostic analysis

Graph descriptors were evaluated as predictors of recognition error.

Analyses included:

- global Spearman correlations;
- structural feature subsets;
- geometry controls;
- graph-quality proxy features;
- stratification by dataset and text level;
- five-fold stratified cross-validation;
- logistic regression for top-quantile high-error detection;
- ROC-AUC;
- PR-AUC;
- top-20% precision.

The diagnostic score is interpreted as sample difficulty, not as direct
gold graph quality.
"""


def make_results(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    desc = e["h1"]["descriptive"]
    image = desc["image_only"]
    graph = desc["graph_recomputed_v3"]

    h2_hc = e["h2"]["hkr_plus_cyrillic"]
    school_old = e["h2"]["old_school_audit"]
    random_v = e["h2"]["random_validation"]

    h3 = e["h3"]["structural_core"]
    corr = e["h3"]["best_global_correlation"]

    runs = e["recognition_cross_evaluation"]["runs"]
    old_old = runs["old_model_old_features"]
    old_new = runs["old_model_new_features"]
    new_old = runs["new_model_old_features"]
    new_new = runs["new_model_new_features"]

    family_rows = []

    for name, row in sorted(e["h1"]["families"].items()):
        family = name.removeprefix("family:")

        family_rows.append(
            f"| `{family}` | "
            f"{pct(row['image_relative_degradation'])} | "
            f"{pct(row['graph_relative_degradation'])} | "
            f"{pct(row['relative_advantage'])} | "
            f"{pct_ci(row['bootstrap']['relative_advantage_ci95'])} | "
            f"{row['permutation']['relative_advantage_one_sided_p']:.6f} | "
            f"{relative_status(row)} |"
        )

    return f"""
# Results

## 1. Recognition performance

| model/checkpoint | feature manifest | CER | WER | exact |
|---|---|---:|---:|---:|
| graph-v2 | old | {fmt(old_old["cer"], 5)} | {fmt(old_old["wer"], 5)} | {fmt(old_old["exact"], 5)} |
| graph-v2 | foreground v3 | {fmt(old_new["cer"], 5)} | {fmt(old_new["wer"], 5)} | {fmt(old_new["exact"], 5)} |
| graph-v3 retrain | old | {fmt(new_old["cer"], 5)} | {fmt(new_old["wer"], 5)} | {fmt(new_old["exact"], 5)} |
| graph-v3 retrain | foreground v3 | {fmt(new_new["cer"], 5)} | {fmt(new_new["wer"], 5)} | {fmt(new_new["exact"], 5)} |

Replacing the old graph features with foreground-v3 features changed the
retained graph-v2 CER by
{signed(float(old_new["cer"]) - float(old_old["cer"]))}.

The new checkpoint remained worse with both feature manifests. Its
degradation was therefore caused primarily by the training run rather
than by foreground repair.

## 2. Descriptive robustness

| model | clean CER | mean distorted CER | absolute delta | relative degradation |
|---|---:|---:|---:|---:|
| image-only | {fmt(image["clean_cer"], 5)} | {fmt(image["mean_distorted_cer"], 5)} | {fmt(image["mean_absolute_delta"], 5)} | {pct(image["mean_relative_degradation"])} |
| graph-vector, recomputed features | {fmt(graph["clean_cer"], 5)} | {fmt(graph["mean_distorted_cer"], 5)} | {fmt(graph["mean_absolute_delta"], 5)} | {pct(graph["mean_relative_degradation"])} |

The graph-vector recognizer has lower proportional degradation but worse
clean and distorted absolute CER.

## 3. Paired corpus robustness

| metric | result |
|---|---:|
| image-only relative degradation | {pct(h1["image_relative_degradation"])} |
| graph relative degradation | {pct(h1["graph_relative_degradation"])} |
| relative advantage | {pct(h1["relative_advantage"])} |
| relative advantage 95% CI | {pct_ci(h1["bootstrap"]["relative_advantage_ci95"])} |
| one-sided permutation p | {h1["permutation"]["relative_advantage_one_sided_p"]:.6f} |
| absolute degradation advantage | {fmt(h1["absolute_advantage"], 5)} |
| absolute advantage 95% CI | {raw_ci(h1["bootstrap"]["absolute_advantage_ci95"])} |
| graph − image distorted CER | {fmt(h1["distorted_cer_gap"], 5)} |

The graph model has a statistically supported relative robustness
advantage. It does not have a positive absolute degradation advantage
and remains worse in absolute distorted CER.

## 4. Robustness by distortion family

| family | image relative | graph relative | advantage | 95% CI | p | verdict |
|---|---:|---:|---:|---:|---:|---|
{os.linesep.join(family_rows)}

Relative robustness is supported for low contrast, additive noise, and
stroke thinning. Blur is inconclusive under the combined confidence-
interval and permutation criterion. Stroke thickening provides no
relative advantage.

## 5. Original H2 diagnostic audit

| subset | n | critical topology error | skeleton follows ink | mean graph quality |
|---|---:|---:|---:|---:|
| HKR + Cyrillic | {h2_hc["n"]} | {pct(h2_hc["critical_topology_error_rate"])} | {pct(h2_hc["skeleton_follows_ink_rate"])} | {fmt(h2_hc["mean_graph_quality_0_3"], 3)} |
| School Notebooks, old preprocessing | {school_old["n"]} | {pct(school_old["critical_topology_error_rate"])} | {pct(school_old["skeleton_follows_ink_rate"])} | {fmt(school_old["mean_graph_quality_0_3"], 3)} |

The School Notebooks failure was localized to foreground extraction.

## 6. Independent random-100 foreground validation

| metric | count | rate |
|---|---:|---:|
| raw good fix | {metric(random_v, "good_fix", "count")}/100 | {pct(metric(random_v, "good_fix"))} |
| partial fix | {metric(random_v, "partial_fix", "count")}/100 | {pct(metric(random_v, "partial_fix"))} |
| bad fix | {metric(random_v, "bad_fix", "count")}/100 | {pct(metric(random_v, "bad_fix"))} |
| strict usable | {metric(random_v, "strict_usable", "count")}/100 | {pct(metric(random_v, "strict_usable"))} |
| real ink erased | {metric(random_v, "real_ink_erased", "count")}/100 | {pct(metric(random_v, "real_ink_erased"))} |
| residual artifact | {metric(random_v, "background_artifact_after", "count")}/100 | {pct(metric(random_v, "background_artifact_after"))} |
| skeleton follows ink | {metric(random_v, "skeleton_follows_ink_after", "count")}/100 | {pct(metric(random_v, "skeleton_follows_ink_after"))} |

The random validation supports `school_dark_auto` for the sampled School
Notebooks test distribution.

## 7. H3 graph diagnostics

| metric | result |
|---|---:|
| best global feature | `{corr.get("feature", "n/a")}` |
| global Spearman r | {fmt(corr.get("spearman_r"))} |
| structural feature set | `{h3.get("feature_set", "n/a")}` |
| subgroup | `{h3.get("group", "n/a")}` |
| n | {h3.get("n", "n/a")} |
| ROC-AUC | {fmt(h3.get("roc_auc"))} |
| PR-AUC | {fmt(h3.get("pr_auc"))} |
| PR-AUC lift | {fmt(h3.get("pr_auc_lift_over_base_rate"))} |
| top-20% precision | {fmt(h3.get("top20_precision"))} |

Individual global descriptors have weak correlations with CER.
Multifeature graph descriptors provide useful but localized high-error
detection.
"""


def make_discussion(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    random_v = e["h2"]["random_validation"]
    h3 = e["h3"]["structural_core"]

    return f"""
# Discussion

## 1. Relative robustness and absolute recognition

The principal positive result is a statistically supported relative
robustness advantage of {pct(h1["relative_advantage"])}. The cluster-
bootstrap interval, {pct_ci(h1["bootstrap"]["relative_advantage_ci95"])},
does not include zero, and the paired permutation test yields
p={h1["permutation"]["relative_advantage_one_sided_p"]:.6f}.

This result must not be interpreted as superior recognition. The graph
model starts from a higher clean CER and remains worse on distorted
images by {fmt(h1["distorted_cer_gap"], 5)} CER.

A weaker model can exhibit smaller proportional degradation because its
initial error is already high. Reporting both relative and absolute
effects is therefore necessary.

## 2. Robustness mechanisms

The strongest relative advantages occur for:

- additive noise;
- reduced contrast;
- thinning of strokes.

These perturbations alter local pixel evidence while leaving part of the
coarse structural organization recoverable.

Stroke thickening provides no advantage. This suggests that graph
descriptors are not uniformly invariant to all morphology changes.

Blur is inconclusive under the strict combined criterion: the
permutation result is positive, but the bootstrap confidence interval
crosses zero.

## 3. Foreground extraction as a structural bottleneck

The original School Notebooks failure illustrates that graph
construction cannot compensate for an incorrect foreground mask.
Background classified as foreground generates artificial components,
skeleton branches, endpoints, and graph edges.

The `school_dark_auto` repair reached a strict usable rate of
{pct(metric(random_v, "strict_usable"))} on an independent random sample.

The remaining {pct(metric(random_v, "real_ink_erased"))} ink-loss rate
and {pct(metric(random_v, "background_artifact_after"))} residual-
artifact rate show that the method is a substantial correction rather
than a universal segmentation solution.

## 4. Representation quality versus fusion utility

Foreground repair visibly improves skeleton and graph plausibility but
does not materially improve HTR.

This separates two questions:

1. Is the structural representation faithful enough for inspection?
2. Does the current recognition architecture use it effectively?

The experiments support the first question more strongly than the
second.

A global graph vector broadcast across all sequence positions may be too
coarse to improve local character recognition. However, further
architecture search is outside the scope of the frozen study because
the current evidence does not justify continued CER-driven tuning.

## 5. Diagnostic role

The strongest H3 result reaches ROC-AUC {fmt(h3.get("roc_auc"))} in
`{h3.get("group", "n/a")}`.

This is useful for ranking difficult samples, prioritizing manual review,
or selecting examples for structural inspection. It is not sufficient
as a universal confidence score.

The weak global correlations indicate that recognition difficulty is
not determined by one graph statistic. It emerges from interactions
between writing style, text level, dataset, preprocessing, and model
behaviour.

## 6. Scientific contribution

The contribution is a controlled empirical characterization of a
visible-stroke structural representation.

The project demonstrates:

- how to construct reproducible graph descriptors from static handwriting;
- how upstream preprocessing failures propagate into graph topology;
- how to validate a dataset-specific foreground repair;
- how to separate relative robustness from absolute recognition quality;
- how to use graph features for localized failure triage;
- why simple global graph fusion does not automatically improve HTR.

The negative recognition result is therefore part of the contribution,
rather than a reason to discard the structural representation.
"""


def make_limitations(e: dict[str, Any]) -> str:
    random_v = e["h2"]["random_validation"]

    return f"""
# Limitations

## 1. Absolute HTR quality

The graph-aware models do not outperform the image-only baseline in
clean or distorted absolute CER.

The robustness result is restricted to relative sensitivity.

## 2. Dependence on the clean baseline

Relative degradation is normalized by clean CER. Since the graph model
starts from a worse baseline, proportional degradation can appear more
favourable even when absolute error remains higher.

The study therefore reports relative advantage, absolute degradation,
and absolute distorted CER separately.

## 3. Synthetic perturbations

The robustness protocol uses controlled synthetic blur, Gaussian noise,
contrast reduction, stroke thinning, and stroke thickening.

It does not reproduce the complete distribution of:

- camera blur;
- JPEG artifacts;
- shadows;
- page curvature;
- bleed-through;
- ink variation;
- mixed illumination;
- real scanning defects.

## 4. Manual H2 audit selection

The original H2 audit subset was selected diagnostically across error
and structural-risk strata.

Its rates characterize failure modes and cannot be interpreted as
population estimates.

## 5. Random School Notebooks validation

The independent validation contains {random_v["n"]} items from one test
split.

It does not establish uniform behaviour across:

- training and validation splits;
- unseen notebook collections;
- different crop-generation procedures;
- different acquisition devices;
- other handwriting datasets.

Annotations were produced by one evaluator. Inter-rater agreement was
not measured.

## 6. Remaining segmentation errors

Foreground v3 removed real ink in
{pct(metric(random_v, "real_ink_erased"))} of the random sample and
retained background artifacts in
{pct(metric(random_v, "background_artifact_after"))}.

The preprocessing method is accepted for the current pipeline but is
not a perfect foreground segmenter.

## 7. No exhaustive graph ground truth

The project does not include complete gold node-edge annotations for
every sample.

Most automated graph-quality descriptors are proxies. Visual audit
supports plausibility but does not establish exact topological
correctness.

## 8. Training variance

The graph-v3 retraining run was worse under both old and repaired feature
manifests.

This identifies a training-run effect but does not quantify full
seed-to-seed variance.

## 9. Localized H3 evidence

The strongest error-detection result is restricted to a particular
dataset and text-level subgroup.

It should not be generalized to the entire mixed corpus.

## 10. Offline structure only

The graph encodes visible static stroke structure.

It does not recover:

- stroke order;
- pen lifts;
- pressure;
- velocity;
- acceleration;
- writer motor dynamics;
- true online trajectory.
"""


def make_conclusion(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    random_v = e["h2"]["random_validation"]
    h3 = e["h3"]["structural_core"]

    return f"""
# Conclusion

This study evaluated canonical visible-stroke graph descriptors as an
intermediate representation for offline handwritten text recognition.

The graph-vector recognizer demonstrated a statistically supported
relative robustness advantage of {pct(h1["relative_advantage"])}, with a
95% cluster-bootstrap interval of
{pct_ci(h1["bootstrap"]["relative_advantage_ci95"])}.

Strong H1 was nevertheless rejected because:

- the absolute degradation advantage was not positive;
- the graph model had worse clean CER;
- the graph model had worse distorted CER;
- the final distorted CER gap was {fmt(h1["distorted_cer_gap"], 5)}.

The structural audit showed that graph plausibility depends strongly on
foreground extraction. The `school_dark_auto` repair achieved a strict
usable rate of {pct(metric(random_v, "strict_usable"))} on an independent
random School Notebooks test sample.

Graph-derived descriptors also provided localized high-error detection,
with a best ROC-AUC of {fmt(h3.get("roc_auc"))}. Their value is therefore
diagnostic rather than universally predictive.

The final contribution is a reproducible structural analysis framework
for:

- visible-stroke graph extraction;
- robustness evaluation;
- preprocessing failure diagnosis;
- structural audit;
- difficult-sample triage.

Current graph-fusion models do not provide superior recognition
accuracy. Future work should focus on localized structural
representations, stronger graph supervision, and broader real-world
degradation evaluation rather than continued tuning of the current
global graph-vector architecture.
"""


def make_reproducibility(e: dict[str, Any]) -> str:
    paths = e["source_paths"]

    return f"""
# Reproducibility

## 1. Canonical evidence inputs

| evidence | path |
|---|---|
| final H1 statistics | `{paths["h1"]}` |
| H2 manual audit | `{paths["h2_manual"]}` |
| random-100 validation | `{paths["h2_random"]}` |
| controlled School foreground conclusion | `{paths["school_controlled"]}` |
| graph-feature cross-evaluation | `{paths["cross_eval"]}` |
| H3 final diagnostic summary | `{paths["h3"]}` |

## 2. Final graph-ready manifest

```text
data/experiments/htr_graph_v1/graph_ready/
tri10k_mixed_school_fg_v3_auto/
````

Expected files:

```text
train.jsonl
val.jsonl
test.jsonl
vocab.json
summary.json
```

## 3. Final retained graph checkpoint

```text
outputs/htr_graph_v1/
tri10k_graph_fusion_v2_lowcap_all/
best.pt
```

The graph-v3 retrained checkpoint is retained only as a negative
controlled experiment.

## 4. Final preprocessing configuration

```text
School Notebooks:
    method: school_dark_auto
    primary threshold: 145
    fallback threshold: 120
    fallback trigger: foreground fraction > 0.35
    minimum connected-object size: 4
```

## 5. Robustness configuration

```text
source samples: 5563
conditions per sample: 15
total joined distorted records: 83445
bootstrap iterations: 5000
permutations: 20000
cluster unit: clean source sample
seed: 20260616
```

## 6. H1 source outputs

```text
outputs/robustness_v2_recomputed/
    graph_vector_v2/
    paired_analysis/
    paired_corpus_v3/
    robustness_graph_modes_v2.json
    robustness_graph_modes_v2.md
```

## 7. H2 source outputs

```text
outputs/h2_gold_audit_v1/
    h2_manual_audit_summary_v2.json
    school_foreground_v3/
    school_foreground_v3_random/
```

## 8. H3 source outputs

```text
outputs/h3_graph_quality_v1/
after_school_fg_v3_auto/
```

## 9. Required environment

The project requires:

* Python;
* PyTorch;
* NumPy;
* Pillow;
* OpenCV;
* scikit-image;
* SciPy;
* scikit-learn.

Exact installed versions should be archived from the active environment:

```bash
python --version
python -m pip freeze > outputs/final_documentation_v3/pip_freeze.txt
```

## 10. Repository state

Before archiving the final experiment:

```bash
git status --short
git rev-parse HEAD
git log -1 --oneline
```

Store the output in:

```text
outputs/final_documentation_v3/repository_state.txt
```

## 11. Determinism notes

Deterministic or fixed-seed components include:

* dataset manifest generation;
* robustness corruption seed;
* random-100 sampling seed;
* bootstrap seed;
* permutation-test seed.

Neural-network training may still vary because of device-specific and
library-level nondeterminism.
"""

def make_tables(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    desc = e["h1"]["descriptive"]
    random_v = e["h2"]["random_validation"]
    h3 = e["h3"]["structural_core"]
    runs = e["recognition_cross_evaluation"]["runs"]

    family_rows = []

    for name, row in sorted(e["h1"]["families"].items()):
        family = name.removeprefix("family:")

        family_rows.append(
            f"| {family} | "
            f"{pct(row['image_relative_degradation'])} | "
            f"{pct(row['graph_relative_degradation'])} | "
            f"{pct(row['relative_advantage'])} | "
            f"{pct_ci(row['bootstrap']['relative_advantage_ci95'])} | "
            f"{relative_status(row)} |"
        )

    return f"""

# Final tables

## Table 1. Hypothesis verdicts

| hypothesis                                 | verdict                         |
| ------------------------------------------ | ------------------------------- |
| Strong H1: better robust HTR system        | rejected                        |
| Partial H1: lower relative sensitivity     | supported                       |
| H2: visible structure preservation         | partially supported             |
| School foreground repair                   | supported on random test sample |
| H3: structural error diagnostics           | localized partial support       |
| Graph fusion improves absolute recognition | not supported                   |

## Table 2. Absolute recognition and robustness

| model        |                                          clean CER |                                          mean distorted CER |                                            relative degradation |
| ------------ | -------------------------------------------------: | ----------------------------------------------------------: | --------------------------------------------------------------: |
| image-only   |          {fmt(desc["image_only"]["clean_cer"], 5)} |          {fmt(desc["image_only"]["mean_distorted_cer"], 5)} |          {pct(desc["image_only"]["mean_relative_degradation"])} |
| graph-vector | {fmt(desc["graph_recomputed_v3"]["clean_cer"], 5)} | {fmt(desc["graph_recomputed_v3"]["mean_distorted_cer"], 5)} | {pct(desc["graph_recomputed_v3"]["mean_relative_degradation"])} |

## Table 3. Primary paired robustness result

| statistic          |                                                     value |
| ------------------ | --------------------------------------------------------: |
| relative advantage |                           {pct(h1["relative_advantage"])} |
| 95% CI             |      {pct_ci(h1["bootstrap"]["relative_advantage_ci95"])} |
| permutation p      | {h1["permutation"]["relative_advantage_one_sided_p"]:.6f} |
| absolute advantage |                        {fmt(h1["absolute_advantage"], 5)} |
| absolute 95% CI    |      {raw_ci(h1["bootstrap"]["absolute_advantage_ci95"])} |
| distorted CER gap  |                         {fmt(h1["distorted_cer_gap"], 5)} |

## Table 4. Robustness families

| family                         | image relative | graph relative | advantage | 95% CI | verdict |
| ------------------------------ | -------------: | -------------: | --------: | -----: | ------- |
| {os.linesep.join(family_rows)} |                |                |           |        |         |

## Table 5. Random School Notebooks validation

| metric               |                                                     count |                                                  rate |
| -------------------- | --------------------------------------------------------: | ----------------------------------------------------: |
| good fix             |                   {metric(random_v, "good_fix", "count")} |                   {pct(metric(random_v, "good_fix"))} |
| partial fix          |                {metric(random_v, "partial_fix", "count")} |                {pct(metric(random_v, "partial_fix"))} |
| bad fix              |                    {metric(random_v, "bad_fix", "count")} |                    {pct(metric(random_v, "bad_fix"))} |
| strict usable        |              {metric(random_v, "strict_usable", "count")} |              {pct(metric(random_v, "strict_usable"))} |
| ink loss             |            {metric(random_v, "real_ink_erased", "count")} |            {pct(metric(random_v, "real_ink_erased"))} |
| residual artifact    |  {metric(random_v, "background_artifact_after", "count")} |  {pct(metric(random_v, "background_artifact_after"))} |
| skeleton follows ink | {metric(random_v, "skeleton_follows_ink_after", "count")} | {pct(metric(random_v, "skeleton_follows_ink_after"))} |

## Table 6. Graph-feature cross-evaluation

| model    |                                old features CER |                      foreground-v3 features CER |                                                                                                 delta |
| -------- | ----------------------------------------------: | ----------------------------------------------: | ----------------------------------------------------------------------------------------------------: |
| graph-v2 | {fmt(runs["old_model_old_features"]["cer"], 5)} | {fmt(runs["old_model_new_features"]["cer"], 5)} | {signed(float(runs["old_model_new_features"]["cer"]) - float(runs["old_model_old_features"]["cer"]))} |
| graph-v3 | {fmt(runs["new_model_old_features"]["cer"], 5)} | {fmt(runs["new_model_new_features"]["cer"], 5)} | {signed(float(runs["new_model_new_features"]["cer"]) - float(runs["new_model_old_features"]["cer"]))} |

## Table 7. H3 diagnostic signal

| feature set                      | subgroup                   |                    n |                  ROC-AUC |                  PR-AUC |                 top-20 precision |
| -------------------------------- | -------------------------- | -------------------: | -----------------------: | ----------------------: | -------------------------------: |
| `{h3.get("feature_set", "n/a")}` | `{h3.get("group", "n/a")}` | {h3.get("n", "n/a")} | {fmt(h3.get("roc_auc"))} | {fmt(h3.get("pr_auc"))} | {fmt(h3.get("top20_precision"))} |
|                              |                            |                      |                          |                         |                                  |
"""

def make_figure_plan(e: dict[str, Any]) -> str:
    return """

# Final figure plan and captions

## Figure 1. End-to-end HI-CSG-R pipeline

### Content

A horizontal pipeline:

```text
grayscale crop
→ foreground extraction
→ skeleton
→ canonical graph descriptors
→ graph-aware HTR / diagnostics
```

### Caption

**Figure 1.** Overview of the HI-CSG-R pipeline. The method converts an
offline grayscale handwriting crop into a deterministic visible-stroke
foreground mask, skeleton, and canonical structural descriptor vector.
The graph represents visible static structure and does not reconstruct
the real pen trajectory.

## Figure 2. School Notebooks preprocessing failure and repair

### Panels

1. original crop;
2. old adaptive foreground;
3. old skeleton;
4. `school_dark_auto` foreground;
5. repaired skeleton.

### Caption

**Figure 2.** Representative School Notebooks foreground-extraction
failure and deterministic repair. The old preprocessing classifies
darker page background as foreground, producing artificial skeleton
structure. `school_dark_auto` suppresses the background while retaining
the visible handwriting.

### Source

```text
outputs/h2_gold_audit_v1/school_foreground_v3/
outputs/h2_gold_audit_v1/school_foreground_v3_random/
```

## Figure 3. Relative robustness by distortion family

### Plot

Grouped bars or point estimates with 95% intervals:

* image-only relative degradation;
* graph-model relative degradation;
* relative advantage interval.

### Caption

**Figure 3.** Relative CER degradation under five synthetic distortion
families. The graph-vector model shows statistically supported relative
advantages for low contrast, additive noise, and stroke thinning. Blur
is inconclusive under the combined criterion, and stroke thickening
shows no advantage.

### Source

```text
outputs/robustness_v2_recomputed/
paired_corpus_v3/paired_corpus_v3.json
```

## Figure 4. Relative robustness versus absolute CER

### Plot

Two-axis or paired-panel visualization:

* relative degradation;
* mean distorted CER.

### Caption

**Figure 4.** Relative robustness does not imply superior absolute
recognition. Although the graph-vector model degrades less
proportionally, its clean and distorted CER remain higher than those of
the image-only baseline.

## Figure 5. Random-100 School Notebooks validation

### Plot

Bar chart with Wilson intervals:

* good fix;
* partial fix;
* strict usable;
* ink loss;
* residual artifact;
* skeleton follows ink.

### Caption

**Figure 5.** Independent validation of `school_dark_auto` on 100
randomly sampled School Notebooks test items. The method achieves high
visual repair and skeleton-following rates while retaining a small
residual ink-loss and background-artifact rate.

## Figure 6. H3 structural high-error detection

### Plot

ROC or precision-ranking plot for the best structural-core subgroup.

### Caption

**Figure 6.** Localized high-error detection using multifeature graph
descriptors. The strongest signal is observed in the HKR word subgroup,
while global individual-feature correlations remain weak.

### Source

```text
outputs/h3_graph_quality_v1/
after_school_fg_v3_auto/
```

## Figure-production rule

All final figures must:

* derive values from machine-readable JSON;
* avoid manually retyping numerical results;
* include sample counts;
* distinguish descriptive and inferential statistics;
* state when intervals are bootstrap or Wilson intervals;
* avoid implying absolute HTR superiority.
  """

def make_claim_matrix(e: dict[str, Any]) -> str:
    safe = "\n".join(
    f"- {claim}" for claim in e["safe_claims"]
    )
    unsafe = "\n".join(
    f"- {claim}" for claim in e["unsafe_claims"]
    )


    return f"""
```

# Final claim matrix

## Frozen thesis claim

> {e["frozen_claim"]}

## Claim status

| claim                                            | status                                 | mandatory qualification                 |
| ------------------------------------------------ | -------------------------------------- | --------------------------------------- |
| Graph-aware HTR is more accurate.                | not supported                          | Absolute CER is worse.                  |
| Graph-vector HTR has lower relative degradation. | supported                              | Restricted to tested distortions.       |
| Graph-vector HTR has lower absolute degradation. | not supported                          | Overall absolute advantage is negative. |
| School foreground v3 repairs extraction.         | supported on sampled test distribution | Report residual failures.               |
| Foreground repair improves recognition.          | not supported                          | Cross-evaluation change is very small.  |
| Graph descriptors detect difficult samples.      | partially supported                    | Signal is subgroup-specific.            |
| Structural risk is graph correctness.            | not supported                          | Treat as difficulty indicator.          |
| Graph is reconstructed pen trajectory.           | false by design                        | Use “visible-stroke structure”.         |

## Safe claims

{safe}

## Claims to avoid

{unsafe}
"""

def make_registry(e: dict[str, Any]) -> str:
    return """

# Experiment registry

| component                   | final status                    | retained artifact                           |
| --------------------------- | ------------------------------- | ------------------------------------------- |
| image-only recognizer       | primary absolute baseline       | image-only clean and robustness outputs     |
| graph-vector v2             | retained graph-aware checkpoint | `tri10k_graph_fusion_v2_lowcap_all/best.pt` |
| gated graph model           | analyzed, not primary           | robustness v1 outputs                       |
| graph-v3 retrain            | rejected checkpoint             | retained as controlled negative result      |
| old School preprocessing    | rejected                        | historical audit only                       |
| border suppression v1       | rejected                        | diagnostic experiment only                  |
| global threshold 145        | candidate only                  | superseded by auto rule                     |
| `school_dark_auto`          | accepted                        | final School foreground method              |
| H1 descriptive robustness   | completed                       | robustness mode comparison                  |
| H1 paired statistics        | completed                       | paired corpus v3                            |
| H2 diagnostic audit         | completed                       | manual audit summary                        |
| H2 random validation        | completed                       | random-100 summary                          |
| H3 diagnostics              | completed                       | after-school-fg-v3 summary                  |
| new HTR architecture search | frozen                          | no further experiments                      |
| graph-fusion CER tuning     | frozen                          | no further experiments                      |

## Final accepted configuration

```text
recognition baseline:
    image-only model

retained graph-aware model:
    graph-vector v2

School Notebooks graph preprocessing:
    school_dark_auto

primary H1 inference:
    paired corpus relative-degradation advantage

primary H2 evidence:
    diagnostic HKR/Cyrillic audit
    + independent School random-100 validation

primary H3 evidence:
    structural-core subgroup high-error detection
```

"""

def make_abstract_ru(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    random_v = e["h2"]["random_validation"]
    h3 = e["h3"]["structural_core"]

    return f"""
```

# Аннотация

В работе исследуется каноническое графовое представление видимой
штриховой структуры для распознавания рукописного текста по статическим
изображениям. Предлагаемое представление не восстанавливает реальную
траекторию движения пера, а описывает воспроизводимую структуру
видимых штрихов, выделенную из офлайн-изображения.

Экспериментальная оценка выполнена на смешанном русско-английском
наборе данных и включает распознавание, анализ устойчивости,
визуальный аудит графов, проверку предобработки и диагностику ошибок.
В испытаниях с 15 типами и уровнями искажений графовая модель показала
преимущество по относительной деградации CER, равное
{pct(h1["relative_advantage"])}, с 95%-м кластерным bootstrap-интервалом
{pct_ci(h1["bootstrap"]["relative_advantage_ci95"])} и односторонним
перестановочным p-значением
{h1["permutation"]["relative_advantage_one_sided_p"]:.6f}.

При этом абсолютная CER графовой модели осталась хуже, поэтому сильная
гипотеза о превосходстве графового распознавателя была отклонена.

Для School Notebooks была выявлена ошибка выделения переднего плана и
разработан детерминированный метод `school_dark_auto`. На независимой
случайной выборке из {random_v["n"]} примеров строгая доля пригодных
результатов составила
{pct(metric(random_v, "strict_usable"))}.

Графовые дескрипторы также показали локализованную ценность для поиска
трудных примеров: лучший результат достиг ROC-AUC
{fmt(h3.get("roc_auc"))}.

Полученные результаты подтверждают ценность графового представления
для анализа устойчивости, проверки предобработки и диагностики отказов,
но не подтверждают повышение абсолютной точности распознавания при
использовании исследованных схем графового слияния.

## Ключевые слова

распознавание рукописного текста; офлайн-почерк; граф штрихов;
скелетизация; структурные признаки; CTC; устойчивость; анализ ошибок.
"""

def make_abstract_en(e: dict[str, Any]) -> str:
    h1 = e["h1"]["overall"]
    random_v = e["h2"]["random_validation"]
    h3 = e["h3"]["structural_core"]

    return f"""
```

# Abstract

This study investigates canonical graph descriptors of visible stroke
structure as an intermediate representation for offline handwritten
text recognition. The representation does not reconstruct the actual
pen trajectory; instead, it captures reproducible visible structure
from static handwriting images.

The experimental evaluation covers recognition, robustness, visual
graph audit, preprocessing validation, and error diagnostics on a mixed
Russian-English handwriting corpus. Across 15 distortion conditions,
the graph-vector model achieved a relative CER degradation advantage of
{pct(h1["relative_advantage"])}, with a 95% paired cluster-bootstrap
interval of {pct_ci(h1["bootstrap"]["relative_advantage_ci95"])} and a
one-sided paired permutation p-value of
{h1["permutation"]["relative_advantage_one_sided_p"]:.6f}.

The graph model nevertheless retained worse absolute CER, and the strong
hypothesis of superior graph-aware recognition was rejected.

A foreground-extraction failure was identified for School Notebooks,
leading to the deterministic `school_dark_auto` repair. Independent
validation on {random_v["n"]} randomly sampled test items yielded a
strict usable rate of {pct(metric(random_v, "strict_usable"))}.

Graph descriptors also showed localized value for high-error detection,
with a best ROC-AUC of {fmt(h3.get("roc_auc"))}.

The results support visible-stroke graph descriptors as tools for
relative robustness analysis, preprocessing validation, and failure
triage, but not as a currently superior route to absolute recognition
accuracy.

## Keywords

handwritten text recognition; offline handwriting; stroke graph;
skeletonization; structural descriptors; CTC; robustness; error
analysis.
"""

def make_full_manuscript(
*,
abstract_ru: str,
abstract_en: str,
methods: str,
results: str,
discussion: str,
limitations: str,
conclusion: str,
) -> str:
    return "\n\n---\n\n".join([
"# Canonical visible-stroke graph descriptors for offline handwritten text recognition",
abstract_ru,
abstract_en,
methods,
results,
discussion,
limitations,
conclusion,
])

# ---------------------------------------------------------------------

# Validation

# ---------------------------------------------------------------------

def validate_package(
out_dir: Path,
expected_files: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []


    for name in expected_files:
        path = out_dir / name

        if not path.exists():
            errors.append(f"missing file: {name}")
            continue

        if path.stat().st_size < 100:
            errors.append(f"file too small: {name}")

    markdown_files = [
        out_dir / name
        for name in expected_files
        if name.endswith(".md")
        and (out_dir / name).exists()
    ]

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in markdown_files
    )

    required_phrases = [
        "Strong H1",
        "relative robustness",
        "absolute CER",
        "school_dark_auto",
        "random",
        "visible",
        "trajectory",
        "Limitations",
        "Reproducibility",
    ]

    for phrase in required_phrases:
        if phrase.lower() not in combined.lower():
            errors.append(
                f"required phrase missing: {phrase}"
            )

    forbidden_phrases = [
        "Strong H1 supported: yes",
        "Graph-aware HTR outperforms image-only HTR.",
        "state-of-the-art recognizer",
        "reconstructs the real pen trajectory",
        "TODO",
        "TBD",
        "FIXME",
    ]

    for phrase in forbidden_phrases:
        if phrase.lower() in combined.lower():
            errors.append(
                f"forbidden phrase found: {phrase}"
            )

    if "partial" not in combined.lower():
        warnings.append(
            "word 'partial' not found in package"
        )

    return {
        "version": PACKAGE_VERSION,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked_files": expected_files,
    }


# ---------------------------------------------------------------------

# Main

# ---------------------------------------------------------------------

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

    source_paths = {
        "h1": args.h1_json,
        "h2_manual": args.h2_manual_json,
        "h2_random": args.h2_random_json,
        "school_controlled": args.school_controlled_json,
        "cross_eval": args.cross_eval_json,
        "h3": args.h3_json,
    }

    h1 = load_json(args.h1_json)
    h2_manual = load_json(args.h2_manual_json)
    h2_random = load_json(args.h2_random_json)
    school_controlled = load_json(
        args.school_controlled_json
    )
    cross_eval = load_json(args.cross_eval_json)
    h3 = load_json(args.h3_json)

    evidence = build_evidence(
        h1=h1,
        h2_manual=h2_manual,
        h2_random=h2_random,
        school_controlled=school_controlled,
        cross_eval=cross_eval,
        h3=h3,
        source_paths=source_paths,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abstract_ru = textwrap.dedent(
        make_abstract_ru(evidence)
    ).strip()
    abstract_en = textwrap.dedent(
        make_abstract_en(evidence)
    ).strip()
    methods = textwrap.dedent(
        make_methods(evidence)
    ).strip()
    results = textwrap.dedent(
        make_results(evidence)
    ).strip()
    discussion = textwrap.dedent(
        make_discussion(evidence)
    ).strip()
    limitations = textwrap.dedent(
        make_limitations(evidence)
    ).strip()
    conclusion = textwrap.dedent(
        make_conclusion(evidence)
    ).strip()

    docs = {
        "00_INDEX.md": make_index(),
        "01_EXECUTIVE_SUMMARY.md": (
            make_executive_summary(evidence)
        ),
        "02_METHODS.md": methods,
        "03_RESULTS.md": results,
        "04_DISCUSSION.md": discussion,
        "05_LIMITATIONS.md": limitations,
        "06_CONCLUSION.md": conclusion,
        "07_REPRODUCIBILITY.md": (
            make_reproducibility(evidence)
        ),
        "08_FINAL_TABLES.md": make_tables(evidence),
        "09_FIGURE_PLAN.md": make_figure_plan(evidence),
        "10_CLAIM_MATRIX.md": (
            make_claim_matrix(evidence)
        ),
        "11_EXPERIMENT_REGISTRY.md": (
            make_registry(evidence)
        ),
        "ABSTRACT_RU.md": abstract_ru,
        "ABSTRACT_EN.md": abstract_en,
        "MANUSCRIPT_FULL.md": make_full_manuscript(
            abstract_ru=abstract_ru,
            abstract_en=abstract_en,
            methods=methods,
            results=results,
            discussion=discussion,
            limitations=limitations,
            conclusion=conclusion,
        ),
    }

    for name, text in docs.items():
        write_text(
            out_dir / name,
            textwrap.dedent(text).strip(),
        )

    write_json(
        out_dir / "evidence_manifest.json",
        evidence,
    )

    expected_files = [
        *docs.keys(),
        "evidence_manifest.json",
    ]

    validation = validate_package(
        out_dir,
        expected_files,
    )

    write_json(
        out_dir / "documentation_validation.json",
        validation,
    )

    checksum_files = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    checksum_text = "\n".join(
        f"{sha256(path)}  {path.name}"
        for path in checksum_files
    )

    write_text(
        out_dir / "SHA256SUMS",
        checksum_text,
    )

    if not validation["valid"]:
        raise RuntimeError(
            "Documentation validation failed:\n"
            + "\n".join(validation["errors"])
        )

    print()
    print("documentation package valid")
    print("output:", out_dir)


if __name__ == "__main__":
    main()
