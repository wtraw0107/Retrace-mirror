| ensemble per set | Q conc. set1 | Q conc. set2 | distinct top-1 | same-label top-5 (score) | same-label top-5 (raw) | corr score | corr raw |
|---|---|---|---|---|---|---|---|
| final only, 3 models (base vs ctrl) | 0.999 | 0.997 | 70 | 0.198 | 0.972 | 0.045 | 0.110 |
| final only, 10 models (scale_base vs scale_ctrl) | 0.945 | 0.947 | 260 | 0.505 | 0.999 | 0.187 | 0.292 |
| quickstart: 3 models x 5 ckpts (ckpt5_base vs ckpt5_ctrl) | 0.305 | 0.316 | 780 | 0.901 | 0.999 | 0.223 | 0.308 |
| epoch 12 only, 3 models (ckpt5) | 0.425 | 0.446 | 729 | 0.631 | 0.982 | 0.067 | 0.112 |
| epoch 15 only, 3 models (ckpt5) | 0.566 | 0.626 | 623 | 0.521 | 0.980 | 0.053 | 0.107 |
| epoch 18 only, 3 models (ckpt5) | 0.812 | 0.831 | 408 | 0.409 | 0.981 | 0.039 | 0.108 |
| epoch 21 only, 3 models (ckpt5) | 0.995 | 0.997 | 131 | 0.236 | 0.975 | 0.037 | 0.111 |
| epoch 23 only, 3 models (ckpt5) | 0.998 | 0.998 | 86 | 0.211 | 0.971 | 0.049 | 0.111 |

set1 = seeds 0-2 (+100s), set2 = independently trained seeds 3-5 (+200s). First 1000 test images; corr = median per-test Pearson over 50k train.

```json
{
 "final only, 3 models (base vs ctrl)": {
  "Q_conc_top1_in_top1pct (set1)": 0.999,
  "Q_conc_top1_in_top1pct (set2)": 0.997,
  "distinct_top1 (set1, of 1000)": 70,
  "Qbar_p50/p99/max (set1)": [
   0.0006085833301767707,
   0.04007789492607117,
   0.3574117124080658
  ],
  "same_label_top5 score (set1)": 0.1976,
  "same_label_top5 raw (set1)": 0.9724,
  "corr score (set1 vs set2)": 0.0452377051115036,
  "corr raw (set1 vs set2)": 0.11048721522092819,
  "score_A_test0 (set1)": 23.65671730041504,
  "A_rank_test0 (set1)": 1
 },
 "final only, 10 models (scale_base vs scale_ctrl)": {
  "Q_conc_top1_in_top1pct (set1)": 0.945,
  "Q_conc_top1_in_top1pct (set2)": 0.947,
  "distinct_top1 (set1, of 1000)": 260,
  "Qbar_p50/p99/max (set1)": [
   0.0008395671611651778,
   0.0364135205745697,
   0.2446853667497635
  ],
  "same_label_top5 score (set1)": 0.5052,
  "same_label_top5 raw (set1)": 0.999,
  "corr score (set1 vs set2)": 0.1870693862438202,
  "corr raw (set1 vs set2)": 0.2917342782020569,
  "score_A_test0 (set1)": 1.601171612739563,
  "A_rank_test0 (set1)": 17
 },
 "quickstart: 3 models x 5 ckpts (ckpt5_base vs ckpt5_ctrl)": {
  "Q_conc_top1_in_top1pct (set1)": 0.305,
  "Q_conc_top1_in_top1pct (set2)": 0.316,
  "distinct_top1 (set1, of 1000)": 780,
  "Qbar_p50/p99/max (set1)": [
   0.030502714216709137,
   0.3204038143157959,
   0.5883544683456421
  ],
  "same_label_top5 score (set1)": 0.9014,
  "same_label_top5 raw (set1)": 0.9992,
  "corr score (set1 vs set2)": 0.2231237292289734,
  "corr raw (set1 vs set2)": 0.3081572651863098,
  "score_A_test0 (set1)": 7.314432144165039,
  "A_rank_test0 (set1)": 24
 },
 "epoch 12 only, 3 models (ckpt5)": {
  "Q_conc_top1_in_top1pct (set1)": 0.425,
  "Q_conc_top1_in_top1pct (set2)": 0.446,
  "distinct_top1 (set1, of 1000)": 729,
  "Qbar_p50/p99/max (set1)": [
   0.06753603368997574,
   0.7912606000900269,
   0.9969075322151184
  ],
  "same_label_top5 score (set1)": 0.6306,
  "same_label_top5 raw (set1)": 0.9824,
  "corr score (set1 vs set2)": 0.06721480190753937,
  "corr raw (set1 vs set2)": 0.11172037571668625,
  "score_A_test0 (set1)": -0.4071264863014221,
  "A_rank_test0 (set1)": 41638
 },
 "epoch 15 only, 3 models (ckpt5)": {
  "Q_conc_top1_in_top1pct (set1)": 0.566,
  "Q_conc_top1_in_top1pct (set2)": 0.626,
  "distinct_top1 (set1, of 1000)": 623,
  "Qbar_p50/p99/max (set1)": [
   0.02307782508432865,
   0.6503926515579224,
   0.9881184697151184
  ],
  "same_label_top5 score (set1)": 0.521,
  "same_label_top5 raw (set1)": 0.9798,
  "corr score (set1 vs set2)": 0.05269947275519371,
  "corr raw (set1 vs set2)": 0.10667074471712112,
  "score_A_test0 (set1)": 11.826682090759277,
  "A_rank_test0 (set1)": 9
 },
 "epoch 18 only, 3 models (ckpt5)": {
  "Q_conc_top1_in_top1pct (set1)": 0.812,
  "Q_conc_top1_in_top1pct (set2)": 0.831,
  "distinct_top1 (set1, of 1000)": 408,
  "Qbar_p50/p99/max (set1)": [
   0.004259506706148386,
   0.369549959897995,
   0.8714192509651184
  ],
  "same_label_top5 score (set1)": 0.4086,
  "same_label_top5 raw (set1)": 0.981,
  "corr score (set1 vs set2)": 0.03915249928832054,
  "corr raw (set1 vs set2)": 0.10812404006719589,
  "score_A_test0 (set1)": 0.17254729568958282,
  "A_rank_test0 (set1)": 8227
 },
 "epoch 21 only, 3 models (ckpt5)": {
  "Q_conc_top1_in_top1pct (set1)": 0.995,
  "Q_conc_top1_in_top1pct (set2)": 0.997,
  "distinct_top1 (set1, of 1000)": 131,
  "Qbar_p50/p99/max (set1)": [
   0.000786125659942627,
   0.07096903771162033,
   0.4110107421875
  ],
  "same_label_top5 score (set1)": 0.2358,
  "same_label_top5 raw (set1)": 0.9748,
  "corr score (set1 vs set2)": 0.03671359643340111,
  "corr raw (set1 vs set2)": 0.11127921938896179,
  "score_A_test0 (set1)": 0.09417769312858582,
  "A_rank_test0 (set1)": 5593
 },
 "epoch 23 only, 3 models (ckpt5)": {
  "Q_conc_top1_in_top1pct (set1)": 0.998,
  "Q_conc_top1_in_top1pct (set2)": 0.998,
  "distinct_top1 (set1, of 1000)": 86,
  "Qbar_p50/p99/max (set1)": [
   0.0006115038995631039,
   0.0412524938583374,
   0.342803955078125
  ],
  "same_label_top5 score (set1)": 0.2114,
  "same_label_top5 raw (set1)": 0.9706,
  "corr score (set1 vs set2)": 0.04853619635105133,
  "corr raw (set1 vs set2)": 0.11122717708349228,
  "score_A_test0 (set1)": 0.37612614035606384,
  "A_rank_test0 (set1)": 1381
 }
}
```
