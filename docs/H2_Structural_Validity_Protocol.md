# H2 Structural Validity Protocol v1

## 1. H2 Formulation

H2:

HI-CSG-R produces structurally usable graph representations of offline handwritten samples, preserving the main visible stroke structures sufficiently for diagnostic analysis, failure taxonomy, and graph-quality estimation.

Russian formulation:

Гипотеза H2: разработанный pipeline HI-CSG-R способен строить структурно пригодное графовое представление видимых штрихов offline-рукописи, сохраняющее основные элементы видимой структуры - компоненты, концы штрихов, соединения, петли, разрывы и неоднозначные области - на уровне, достаточном для диагностического анализа качества рукописных данных.

## 2. What H2 Does Not Claim

H2 does not claim:

- exact pen trajectory recovery;
- topology-perfect graph reconstruction;
- OCR improvement;
- robustness under distortions;
- universal validity on every handwriting style or crop;
- that a graph file existing is enough evidence.

H2 claims only structural usability of the visible-stroke representation for analysis and diagnostics.

## 3. Evaluation Layers

Layer A: automatic coverage.

- samples processed;
- foreground extracted;
- skeleton produced;
- graph produced;
- features exported;
- warnings produced;
- overlays generated.

Layer A is required but cannot confirm H2 alone.

Layer B: manual structural usability review.

This is the primary H2 evidence. Review 150-250 samples and annotate whether foreground, skeleton, graph topology, components, endpoints, junctions, and loops are usable for diagnostics.

Layer C: strict gold subset.

On 50-80 samples, optionally annotate endpoint, junction, loop, broken-connection, and false-bridge quality more strictly. This layer strengthens H2 but is not required for the first closure if Layer B is complete and stratified.

## 4. Sampling Plan

Recommended review subset:

- 60 Cyrillic Handwriting / HKR Words;
- 60 School Notebooks Clean;
- 30 hard distorted / robustness cases;
- 30 mixed difficult cases: loops, digits, short words, connected writing, noisy foreground.

Alternative stronger subset:

- 60 Cyrillic;
- 60 HKR;
- 60 School;
- 30 hard structural distortions;
- 30 numeric/mixed/ambiguous.

The review must include hard cases. A clean-only subset is not acceptable H2 evidence.

## 5. Review Labels

Required columns:

- `sample_id`
- `dataset`
- `image_path`
- `overlay_path`
- `foreground_ok`
- `skeleton_usable`
- `graph_usable`
- `components_reasonable`
- `endpoints_reasonable`
- `junctions_reasonable`
- `loops_preserved`
- `severe_topology_error`
- `usable_for_diagnostics`
- `failure_type`
- `notes`

Boolean fields accept `1/0`, `yes/no`, `true/false`, `да/нет`.

`loops_preserved` may be `NA` when no visible loop is present.

## 6. Failure Taxonomy

Use exactly one primary `failure_type` where possible:

- `none`
- `foreground_loss`
- `background_noise`
- `broken_strokes`
- `false_bridges`
- `lost_loop`
- `false_junctions`
- `overbranching`
- `merged_components`
- `split_components`
- `bad_skeleton`
- `uncertain_cursive`
- `other`

If multiple failures are present, choose the dominant one and explain the secondary issue in `notes`.

## 7. Label Semantics

`foreground_ok = 1` if the main visible ink is preserved and false foreground is not dominant.

`skeleton_usable = 1` if the skeleton follows the main visible strokes sufficiently for structural analysis.

`graph_usable = 1` if graph nodes/edges visually reflect the main stroke structure well enough for analysis.

`components_reasonable = 1` if connected components broadly match visible connected ink regions.

`endpoints_reasonable = 1` if stroke endings are mostly represented without dominant false endpoints.

`junctions_reasonable = 1` if joins/crossings are represented without dominant false or missing junctions.

`loops_preserved = 1` if visible loops are preserved when loops exist. Use `NA` if there is no visible loop.

`severe_topology_error = 1` if topology is dominated by false bridges, broken strokes, lost loops, merged unrelated components, or severe overbranching.

`usable_for_diagnostics = 1` if the graph can still be used to diagnose handwriting/foreground/topology quality, even if it is not perfect.

## 8. Acceptance Thresholds

H2 is minimally supported if:

- `foreground_ok >= 90%`
- `skeleton_usable >= 85%`
- `graph_usable >= 80%`
- `usable_for_diagnostics >= 80%`
- `severe_topology_error <= 20%`

H2 is strongly supported if:

- `foreground_ok >= 95%`
- `skeleton_usable >= 90%`
- `graph_usable >= 85%`
- `usable_for_diagnostics >= 85%`
- `severe_topology_error <= 15%`

If one dataset is substantially worse, report it explicitly instead of averaging it away.

## 9. Reporting Rules

The H2 closure report must include:

- overall rates;
- dataset breakdown;
- failure taxonomy;
- good/bad/ambiguous examples;
- explicit limitations;
- statement that H2 is structural usability, not OCR improvement and not exact topology recovery.

Invalid H2 arguments:

- graph files created, therefore H2 confirmed;
- overlays look good informally, therefore H2 confirmed;
- one clean dataset passes, therefore H2 confirmed;
- 100% `graph_ok` without hard cases or failure taxonomy.
