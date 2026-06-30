# Publication Hardening Plan v1

## Objective

Raise the HI-CSG-R study from a strengthened thesis/workshop package to a defensible publication-level empirical package.

## Minimum Required Work

1. Full from-scratch same-size augmentation controls over three seeds.
   - Seeds: 42, 43, 44.
   - Variants:
     - `tri10k_base`: original 30k mixed train set.
     - `line_context_10k`: original 30k plus 9,998 natural-line context crops.
     - `random_crops_10k_control`: original 30k plus 10,000 ordinary crop samples.
     - `school_words_10k_control`: original 30k plus 10,000 School word crops without line context.
   - Shared protocol:
     - CRNN-CTC architecture.
     - 80 epochs.
     - batch size 16.
     - lr 0.0005.
     - weight decay 0.0001.
     - blank penalty schedule -2.0 to -0.4.
     - no AMP for compatibility with existing primary runs.
   - Required output:
     - one best checkpoint per variant/seed;
     - full test predictions per variant/seed;
     - seed-level aggregate table;
     - paired comparisons with confidence intervals.

2. Normalized decoding audit.
   - Re-evaluate all already trained base and line-context seed checkpoints with one fixed test-time decoding protocol.
   - Use fixed `blank_logit_penalty=-0.4` to avoid mixed penalty selection across old summaries.
   - Controls must be evaluated with the same fixed penalty.

3. External strong baseline.
   - Primary target: pretrained TrOCR handwriting model evaluated on the same test split.
   - Required output:
     - predictions JSONL;
     - aggregate CER/WER/exact;
     - grouped metrics by dataset/category;
     - clear statement whether this is zero-shot or fine-tuned.
   - Publication limitation:
     - zero-shot TrOCR is an external pretrained baseline, but a strong journal-level comparison still requires a fine-tuned transformer/HTR baseline under the same train/test protocol.

4. Data leakage and integrity audits.
   - Verify sample-id split overlap.
   - Verify train duplicate ids.
   - Verify OOV and empty text rows.
   - If metadata permits, add writer/page overlap and visual near-duplicate audits.

5. Statistical analysis.
   - Report per-seed mean and standard deviation.
   - Report paired bootstrap confidence intervals on per-sample CER deltas.
   - Separate overall, dataset-wise, and School-only effects.
   - Do not claim universal improvement if any dataset degrades.

6. Structural component evidence.
   - Keep structural claims diagnostic-only unless a stricter topology benchmark is added.
   - Required upgrades for strong publication:
     - inter-annotator agreement;
     - endpoint/junction/topology metrics;
     - pixel-level foreground/skeleton comparison if gold masks exist.

7. Reproducibility package.
   - Archive exact code state.
   - Archive commands, config, seed, environment, checkpoints, and manifest summaries.
   - Produce a clean repository commit or immutable artifact bundle before submission.

## Claim Rules

Allowed only if full controls confirm it:

- Natural-line context improves recognition beyond same-size ordinary crop augmentation.
- The effect is strongest on School Notebooks.

Allowed now with strict wording:

- The previous 3-seed result supports natural-line context augmentation under the original CRNN-CTC protocol.
- Diagnostic controls suggest the effect is not explained only by adding more crop samples.

Forbidden:

- HI-CSG-R is SOTA.
- Graph topology or writing trajectory is recovered.
- Structural gold proves exact graph correctness.
- Zero-shot TrOCR is a fully fair substitute for a fine-tuned external baseline.
