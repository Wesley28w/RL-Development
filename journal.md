# Journal Results

Consolidated from `logs/journal`. n = 16 seeds (42-57) per condition.

| Env. | Condition | IQM | Mean | Median | Std. | Min | Max | 95% CI P(X>Y) | Cliff's d | Perm. p | B-M p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cabinet | Baseline | 0.7199 | 0.6199 | 0.7255 | 0.3072 | 0.0000 | 0.8905 | - | - | - | - |
|  | ACES | 0.8358 | 0.8200 | 0.8302 | 0.0799 | 0.5625 | 0.9099 | [0.496, 0.875] | +0.398 | 0.0100 \* | 0.0543 |
| Factory | Baseline | 0.9109 | 0.9101 | 0.9114 | 0.0106 | 0.8893 | 0.9255 | - | - | - | - |
|  | ACES | 0.9181 | 0.9175 | 0.9189 | 0.0096 | 0.9002 | 0.9338 | [0.504, 0.863] | +0.391 | 0.0494 \* | 0.0469 \* |
| Lift | Baseline | 0.3440 | 0.4216 | 0.1091 | 0.4729 | 0.0000 | 0.9990 | - | - | - | - |
|  | ACES | 0.9969 | 0.8232 | 0.9976 | 0.3657 | 0.0012 | 0.9995 | [0.602, 0.918] | +0.555 | 0.0133 \* | 0.0021 \* |

All comparisons are **unpaired**: the curriculum consumes additional RNG draws at every reset, so a shared seed does not produce matched runs (measured correlation r = 0.53 / 0.09 / 0.06 for Cabinet / Factory / Lift).

Cliff's d = 2*P(X>Y) - 1. 95% CI is the unpaired bootstrap CI on P(X>Y) (B = 10,000 resamples). Perm. p is an unpaired permutation test on the difference of means (100,000 resamples); B-M p is Brunner-Munzel. \* p < 0.05.

Metric: Cabinet and Lift use the mean of the final 100 iterations; Factory uses each seed's peak over its final 25 logged steps.
