# School notebooks border suppression audit — v1

## 1. Purpose

This is a preprocessing sanity check for school-notebooks samples where manual audit identified crop/border artifacts being binarized as foreground.

## 2. Aggregate

| metric | value |
|---|---:|
| n | 23 |
| mean removed components | 0.609 |
| mean removed area frac | 0.10670 |
| mean old fg fraction | 0.38554 |
| mean new fg fraction | 0.27884 |
| mean old skeleton fraction | 0.06723 |
| mean new skeleton fraction | 0.04770 |

## 3. Interpretation rule

If the new binary/skeleton removes crop-border components without erasing real handwriting, then school-notebooks failures should be reported as fixable preprocessing artifacts.

If it erases real handwriting, the rule is too aggressive and should not be integrated.