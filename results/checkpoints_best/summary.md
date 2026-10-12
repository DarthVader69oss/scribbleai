# Evaluation of `/content/drive/MyDrive/scribbleai/checkpoints/best.pt` (epoch 16)

## test (2000 samples)
| decoder | cer | wer | line_acc | total_ms_per_line |
|---|---|---|---|---|
| greedy | 0.0533 | 0.2329 | 0.5490 | 159.8501 |
| beam | 0.0532 | 0.2326 | 0.5490 | 164.9602 |
| beam_lm | 0.0254 | 0.1065 | 0.7505 | 168.4476 |
| beam_lm_spell | 0.0225 | 0.0815 | 0.8065 | 167.1537 |

Top confusions (true → predicted): `∅`→` ` (140), `n`→`u` (128), `o`→`a` (112), `t`→`s` (108), `n`→`r` (102), `r`→`s` (49), `h`→`n` (48), `t`→`r` (44), `r`→`x` (44), `u`→`n` (43), `w`→`m` (33), `p`→`n` (33)

Worst lines:
- `%` → `?,` (CER 2.00)
- `v` → `V` (CER 1.00)
- `a` → `U` (CER 1.00)
- `u` → `h` (CER 1.00)
- `4` → `H` (CER 1.00)

## uji_chars_test (2000 samples)
| decoder | cer | wer | line_acc | total_ms_per_line |
|---|---|---|---|---|
| greedy | 0.2880 | 0.2800 | 0.7205 | 18.6794 |
| beam | 0.2885 | 0.2805 | 0.7200 | 18.9605 |

Top confusions (true → predicted): `0`→`O` (18), `ó`→`Ó` (17), `o`→`O` (16), `v`→`V` (14), `Y`→`y` (14), `P`→`p` (14), `Z`→`z` (13), `x`→`X` (12), `W`→`w` (11), `'`→`I` (11), `z`→`Z` (9), `S`→`s` (9)

Worst lines:
- `"` → `) )` (CER 3.00)
- `H` → `8<` (CER 2.00)
- `d` → `aL` (CER 2.00)
- `"` → `I ` (CER 2.00)
- `q` → `9.` (CER 2.00)

## inhouse_test (294 samples)
| decoder | cer | wer | line_acc | total_ms_per_line |
|---|---|---|---|---|
| greedy | 0.4692 | 1.2586 | 0.1156 | 90.3304 |
| beam | 0.4697 | 1.2586 | 0.1156 | 94.3936 |
| beam_lm | 0.3866 | 0.8391 | 0.2959 | 98.6027 |
| beam_lm_spell | 0.3884 | 0.7947 | 0.3537 | 100.8842 |

Top confusions (true → predicted): `∅`→` ` (384), ` `→`∅` (74), `e`→`∅` (59), `n`→`∅` (41), `t`→`∅` (38), `o`→`∅` (29), ` `→`"` (28), `r`→`∅` (27), `a`→`∅` (25), ` `→`ª` (24), `e`→`c` (23), `c`→`∅` (21)

Worst lines:
- `fox` → `b o >L` (CER 1.67)
- `Pack` → `p a ( k` (CER 1.25)
- `box` → `b o a` (CER 1.00)
- `dozen` → `d o g e n` (CER 1.00)
- `liquor` → `l i g u o r` (CER 1.00)

## Beam-width sweep (synthetic test subset)
| mode | beam | cer | wer | decode_ms |
|---|---|---|---|---|
| beam | 1 | 0.0570 | 0.2316 | 1.5833 |
| beam | 5 | 0.0573 | 0.2316 | 3.3599 |
| beam | 10 | 0.0573 | 0.2316 | 4.3037 |
| beam | 25 | 0.0572 | 0.2316 | 9.2068 |
| beam_lm | 1 | 0.0385 | 0.1561 | 1.1671 |
| beam_lm | 5 | 0.0279 | 0.1153 | 3.3863 |
| beam_lm | 10 | 0.0268 | 0.1103 | 5.9588 |
| beam_lm | 25 | 0.0264 | 0.1093 | 18.0698 |

## Generalisation gap (in-house CER − synthetic test CER)
- greedy: +41.6 points
- beam: +41.7 points
- beam_lm: +36.1 points
- beam_lm_spell: +36.6 points
