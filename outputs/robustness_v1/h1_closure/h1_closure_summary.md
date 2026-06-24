# H1 Closure Report

## Model Degradation Summary

| model | clean CER | mean distorted CER | mean absolute ΔCER | mean relative degradation | visual rel deg | structural rel deg |
|---|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.0822 | 0.1136 | 0.0314 | 0.3820 | 0.3624 | 0.4113 |
| `graph_vector_v2` | 0.1397 | 0.1697 | 0.0300 | 0.2148 | 0.1583 | 0.2995 |
| `gated_v2_dist` | 0.1438 | 0.1724 | 0.0287 | 0.1994 | 0.1533 | 0.2686 |

## Condition Comparison

| condition | image-only CER | graph-vector CER | gated CER | best model | graph-vector - image-only | gated - image-only |
|---|---:|---:|---:|---|---:|---:|
| `clean` | 0.0822 | 0.1397 | 0.1438 | `image_only` | 0.0575 | 0.0615 |
| `blur_mild` | 0.0857 | 0.1478 | 0.1507 | `image_only` | 0.0621 | 0.0649 |
| `blur_medium` | 0.0965 | 0.1618 | 0.1634 | `image_only` | 0.0653 | 0.0669 |
| `blur_strong` | 0.1283 | 0.1950 | 0.1957 | `image_only` | 0.0668 | 0.0675 |
| `noise_mild` | 0.0845 | 0.1416 | 0.1449 | `image_only` | 0.0571 | 0.0605 |
| `noise_medium` | 0.1042 | 0.1503 | 0.1566 | `image_only` | 0.0461 | 0.0524 |
| `noise_strong` | 0.1611 | 0.1757 | 0.1886 | `image_only` | 0.0146 | 0.0276 |
| `low_contrast_mild` | 0.0965 | 0.1477 | 0.1519 | `image_only` | 0.0512 | 0.0553 |
| `low_contrast_medium` | 0.1145 | 0.1595 | 0.1627 | `image_only` | 0.0450 | 0.0482 |
| `low_contrast_strong` | 0.1370 | 0.1768 | 0.1776 | `image_only` | 0.0398 | 0.0406 |
| `thin_strokes_mild` | 0.0947 | 0.1561 | 0.1575 | `image_only` | 0.0614 | 0.0628 |
| `thin_strokes_medium` | 0.0947 | 0.1561 | 0.1575 | `image_only` | 0.0614 | 0.0628 |
| `thin_strokes_strong` | 0.2233 | 0.2970 | 0.2917 | `image_only` | 0.0738 | 0.0684 |
| `thick_strokes_mild` | 0.0869 | 0.1502 | 0.1519 | `image_only` | 0.0633 | 0.0650 |
| `thick_strokes_medium` | 0.0869 | 0.1502 | 0.1519 | `image_only` | 0.0633 | 0.0650 |
| `thick_strokes_strong` | 0.1098 | 0.1796 | 0.1837 | `image_only` | 0.0698 | 0.0739 |

## Paired Sign / Bootstrap

| condition | model | wins | losses | ties | mean ΔCER | CI95 |
|---|---|---:|---:|---:|---:|---:|
| `clean` | `graph_vector_v2` | 504 | 2148 | 2911 | 0.0648 | [0.0607, 0.0688] |
| `clean` | `gated_v2_dist` | 517 | 2253 | 2793 | 0.0688 | [0.0648, 0.0729] |
| `blur_mild` | `graph_vector_v2` | 480 | 2233 | 2850 | 0.0694 | [0.0655, 0.0734] |
| `blur_mild` | `gated_v2_dist` | 496 | 2307 | 2760 | 0.0721 | [0.0679, 0.0762] |
| `blur_medium` | `graph_vector_v2` | 528 | 2324 | 2711 | 0.0729 | [0.0688, 0.0771] |
| `blur_medium` | `gated_v2_dist` | 500 | 2376 | 2687 | 0.0746 | [0.0704, 0.0789] |
| `blur_strong` | `graph_vector_v2` | 613 | 2477 | 2473 | 0.0751 | [0.0707, 0.0795] |
| `blur_strong` | `gated_v2_dist` | 602 | 2490 | 2471 | 0.0756 | [0.0710, 0.0800] |
| `noise_mild` | `graph_vector_v2` | 532 | 2163 | 2868 | 0.0654 | [0.0613, 0.0693] |
| `noise_mild` | `gated_v2_dist` | 510 | 2275 | 2778 | 0.0684 | [0.0642, 0.0724] |
| `noise_medium` | `graph_vector_v2` | 670 | 2078 | 2815 | 0.0565 | [0.0521, 0.0608] |
| `noise_medium` | `gated_v2_dist` | 626 | 2206 | 2731 | 0.0618 | [0.0573, 0.0664] |
| `noise_strong` | `graph_vector_v2` | 1030 | 1978 | 2555 | 0.0304 | [0.0245, 0.0365] |
| `noise_strong` | `gated_v2_dist` | 905 | 2194 | 2464 | 0.0427 | [0.0367, 0.0483] |
| `low_contrast_mild` | `graph_vector_v2` | 607 | 2177 | 2779 | 0.0626 | [0.0583, 0.0670] |
| `low_contrast_mild` | `gated_v2_dist` | 595 | 2237 | 2731 | 0.0646 | [0.0603, 0.0690] |
| `low_contrast_medium` | `graph_vector_v2` | 677 | 2187 | 2699 | 0.0589 | [0.0544, 0.0632] |
| `low_contrast_medium` | `gated_v2_dist` | 683 | 2237 | 2643 | 0.0599 | [0.0554, 0.0644] |
| `low_contrast_strong` | `graph_vector_v2` | 771 | 2246 | 2546 | 0.0582 | [0.0534, 0.0628] |
| `low_contrast_strong` | `gated_v2_dist` | 827 | 2260 | 2476 | 0.0559 | [0.0512, 0.0607] |
| `thin_strokes_mild` | `graph_vector_v2` | 578 | 2261 | 2724 | 0.0691 | [0.0646, 0.0734] |
| `thin_strokes_mild` | `gated_v2_dist` | 579 | 2326 | 2658 | 0.0714 | [0.0672, 0.0759] |
| `thin_strokes_medium` | `graph_vector_v2` | 578 | 2261 | 2724 | 0.0691 | [0.0648, 0.0732] |
| `thin_strokes_medium` | `gated_v2_dist` | 579 | 2326 | 2658 | 0.0714 | [0.0670, 0.0759] |
| `thin_strokes_strong` | `graph_vector_v2` | 815 | 2686 | 2062 | 0.0851 | [0.0798, 0.0904] |
| `thin_strokes_strong` | `gated_v2_dist` | 822 | 2635 | 2106 | 0.0830 | [0.0778, 0.0882] |
| `thick_strokes_mild` | `graph_vector_v2` | 497 | 2234 | 2832 | 0.0716 | [0.0675, 0.0757] |
| `thick_strokes_mild` | `gated_v2_dist` | 466 | 2310 | 2787 | 0.0735 | [0.0695, 0.0777] |
| `thick_strokes_medium` | `graph_vector_v2` | 497 | 2234 | 2832 | 0.0716 | [0.0674, 0.0759] |
| `thick_strokes_medium` | `gated_v2_dist` | 466 | 2310 | 2787 | 0.0735 | [0.0695, 0.0777] |
| `thick_strokes_strong` | `graph_vector_v2` | 602 | 2446 | 2515 | 0.0811 | [0.0763, 0.0856] |
| `thick_strokes_strong` | `gated_v2_dist` | 543 | 2486 | 2534 | 0.0827 | [0.0781, 0.0872] |

## Gate Distribution

Existing gated outputs store only condition-level `gate_mean`; median/p10/p90/max are unavailable without re-running eval with per-sample or per-pixel gate logging.

| condition | gate mean | gate median | gate p10 | gate p90 | gate max |
|---|---:|---:|---:|---:|---:|
| `clean` | 0.0685 | n/a | n/a | n/a | n/a |
| `blur_mild` | 0.0691 | n/a | n/a | n/a | n/a |
| `blur_medium` | 0.0691 | n/a | n/a | n/a | n/a |
| `blur_strong` | 0.0693 | n/a | n/a | n/a | n/a |
| `noise_mild` | 0.0663 | n/a | n/a | n/a | n/a |
| `noise_medium` | 0.0647 | n/a | n/a | n/a | n/a |
| `noise_strong` | 0.0642 | n/a | n/a | n/a | n/a |
| `low_contrast_mild` | 0.0709 | n/a | n/a | n/a | n/a |
| `low_contrast_medium` | 0.0721 | n/a | n/a | n/a | n/a |
| `low_contrast_strong` | 0.0730 | n/a | n/a | n/a | n/a |
| `thin_strokes_mild` | 0.0690 | n/a | n/a | n/a | n/a |
| `thin_strokes_medium` | 0.0690 | n/a | n/a | n/a | n/a |
| `thin_strokes_strong` | 0.0694 | n/a | n/a | n/a | n/a |
| `thick_strokes_mild` | 0.0696 | n/a | n/a | n/a | n/a |
| `thick_strokes_medium` | 0.0696 | n/a | n/a | n/a | n/a |
| `thick_strokes_strong` | 0.0702 | n/a | n/a | n/a | n/a |

## Conclusion

H1 is not confirmed for the current implementation. The image-only model remains the practical winner across clean and distorted conditions. Graph-aware variants sometimes show lower relative degradation because their clean CER is already much worse, but they do not provide practical distorted-CER wins. Gated v2 also does not show evidence of systematic graph-branch activation.
