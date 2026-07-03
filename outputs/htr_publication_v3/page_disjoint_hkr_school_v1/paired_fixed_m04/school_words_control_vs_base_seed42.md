# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1135 | 808 | 2057 | 0.1429 | 0.1211 | -0.0217 | [-0.0265, -0.0171] | 0.4617 | 0.4126 | -0.0491 | 0.4135 | 0.4585 | 0.0450 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 635 | 515 | 850 | 0.1269 | 0.1182 | -0.0087 | 0.4410 | 0.4085 | -0.0325 | 0.3025 | 0.3280 | 0.0255 |
| `school_notebooks_clean` | 2000 | 500 | 293 | 1207 | 0.1589 | 0.1241 | -0.0348 | 0.4825 | 0.4168 | -0.0657 | 0.5245 | 0.5890 | 0.0645 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 83 | 36 | 316 | 0.2172 | 0.1464 | -0.0709 |
| `4-6` | 849 | 236 | 113 | 500 | 0.1613 | 0.1254 | -0.0359 |
| `7-10` | 1006 | 251 | 197 | 558 | 0.1375 | 0.1249 | -0.0126 |
| `11+` | 1710 | 565 | 462 | 683 | 0.1180 | 0.1104 | -0.0076 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0431, -0.0268]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 500 | 293 | 1207 | 0.1589 | 0.1241 | -0.0348 | 0.5245 | 0.5890 | 0.0645 |
