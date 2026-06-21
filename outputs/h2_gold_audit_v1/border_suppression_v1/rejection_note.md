# Border suppression v1 rejection note

## Verdict

Border suppression v1 is rejected.

## Reason

Manual inspection showed that the rule is not reliable:

- in many samples it does not visibly remove the crop/border artifact;
- in some samples it removes nearly all foreground, including handwriting;
- therefore it cannot be integrated into the graph preprocessing pipeline.

## Evidence

Aggregate metrics showed a drop in foreground and skeleton fractions, but this drop is not a valid improvement signal. Several samples had `new_fg_fraction` near zero, indicating over-removal rather than successful cleanup.

## Methodological interpretation

The school-notebooks failures observed in H2 are best treated as upstream crop/background/binarization artifacts, not as canonical graph topology failures.

A proper fix would require dataset-specific preprocessing, such as better crop generation, background normalization, or explicit page-border removal before binarization. This is outside the current H2-v1 graph preservation claim.

## Decision

Do not use border suppression v1 in any reported graph-quality results.

For H2-v1:

- report HKR and Cyrillic graph preservation separately;
- report school-notebooks as a preprocessing failure mode;
- do not aggregate school-notebooks into a single graph-topology failure rate.