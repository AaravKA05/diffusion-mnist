# Linear vs cosine noise schedule (conditional model)

Classifier label accuracy, 100 samples per digit per seed, same starting noise for both schedules. Mean over seeds; the last column is the per-seed difference (cosine minus linear, in points).

alpha_bar_T: linear 0.0175, cosine 1.52e-08

| DDIM steps | w | Linear | Cosine | Cosine - linear, per seed |
|---|---|---|---|---|
| 10 | 1.0 | 89.8% | 90.9% | +1.1, +0.1, +1.9 |
| 10 | 1.2 | 95.4% | 96.2% | +0.5, +1.4, +0.7 |
| 20 | 1.0 | 89.0% | 92.5% | +3.2, +4.2, +3.3 |
| 20 | 1.2 | 94.1% | 97.0% | +2.3, +3.8, +2.6 |
| 50 | 1.0 | 88.0% | 93.2% | +5.5, +5.1, +4.8 |
| 50 | 1.2 | 91.6% | 96.1% | +4.8, +4.8, +4.1 |

Per-digit accuracy at DDIM 20 steps, highest w, mean over seeds:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear | 94 | 91 | 94 | 96 | 93 | 94 | 97 | 93 | 92 | 98 |
| Cosine | 98 | 97 | 97 | 98 | 97 | 96 | 98 | 97 | 94 | 98 |
