# Running experiments

Set `partition=` for slurm

## Predicted Mentions & Gold Topics

Span scorer:

```bash
# slurm
sh bin/launch_span_scorer.sh configs/mixed_span_scorer1_0.40.json $partition 32000
# single machine
sh bin/run_span_scorer.sh configs/mixed_span_scorer1_0.40.json
```

Pairwise:

```bash
# slurm
sh bin/launch_pairwise.sh configs/mixed_pairwise1_0.40.json $partition 32000
# single machine
sh bin/launch_pairwise.sh configs/mixed_pairwise1_0.40.json
```

Tuning threshold:

```bash
# slurm
sh bin/launch_tuned_threshold.sh configs/mixed_clustering1_0.40.json $partition 32000
# single machine
sh bin/run_tuned_threshold.sh configs/mixed_clustering1_0.40.json
```

```bash
# slurm
sh bin/launch_find_best_model.sh models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40 mixed $partition 32000
# single machine
sh bin/run_find_best_model.sh models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40 mixed
```

Take the best model from the log output, e.g.

```bash
tail -1  logs/run_scorer/mixed_0.40/2021-03-07-08-32-37-501471096/log.log
('model_3_dev_mixed_average_0.65_topic_level.conll', 45.65803398409008)
```

Set this in [configs/mixed_clustering1_0.40_test.json](configs/mixed_clustering1_0.40_test.json)
```
  "model_num": 3,
  "threshold": 0.65,
```

Running test:
```bash
# slurm
sh bin/launch_predict.sh configs/mixed_clustering1_0.40_test.json $partition 32000
# single machine
sh bin/run_predict.sh configs/mixed_clustering1_0.40_test.json $partition 32000
```

Scoring test:

```bash
git clone git@github.com:conll/reference-coreference-scorers.git
```

```bash
./scorer.pl  all ../data/ecb/gold/test_mixed_topic_level.conll ../models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40/test_mixed_average_0.65_model_3_topic_level.conll > ../models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40/test_mixed_average_0.65_model_3_topic_level.conll.score
```

```bash
grep -i -e metric -e Coref ../models/pairwise_scorer1/topic_level_predicted_mentions/mixed_0.40/test_mixed_average_0.65_model_3_topic_level.conll.score
version: 8.01 /mnt/nfs/scratch1/nmonath/coref_public/reference-coreference-scorers/lib/CorScorer.pm
METRIC muc:
Coreference: Recall: (1083 / 2420) 44.75%       Precision: (1083 / 1890) 57.3%  F1: 50.25%
METRIC bcub:
Coreference: Recall: (799.806702468333 / 2800) 28.56%   Precision: (909.739533996217 / 2250) 40.43%     F1: 33.47%
METRIC ceafm:
Coreference: Recall: (1024 / 2800) 36.57%       Precision: (1024 / 2250) 45.51% F1: 40.55%
METRIC ceafe:
Coreference: Recall: (119.005559571695 / 380) 31.31%    Precision: (119.005559571695 / 360) 33.05%      F1: 32.16%
METRIC blanc:
Coreference:
Coreference links: Recall: (6335 / 24784) 25.56%        Precision: (6335 / 16236) 39.01%        F1: 30.88%
Non-coreference links: Recall: (110607 / 393377) 28.11% Precision: (110607 / 271380) 40.75%     F1: 33.27%
```


## Gold Mentions & Gold Topics

Pairwise:

```bash
# slurm
sh bin/launch_pairwise.sh configs/gold_mixed_pairwise1_0.40.json $partition 32000
# single machine
sh bin/launch_pairwise.sh configs/gold_mixed_pairwise1_0.40.json
```

Tuning threshold:
```
# slurm
sh bin/launch_tuned_threshold.sh configs/gold_mixed_clustering1_0.40.json $partition 32000
# single machine
sh bin/run_tuned_threshold.sh configs/gold_mixed_clustering1_0.40.json
```

```bash
# slurm
sh bin/launch_find_best_model.sh models/pairwise_scorer1/topic_level_gold_mentions/mixed_0.40 mixed $partition 32000
# single machine
sh bin/run_find_best_model.sh models/pairwise_scorer1/topic_level_gold_mentions/mixed_0.40 mixed
```

```bash
tail -1 logs/run_scorer/mixed_0.40/2021-03-07-09-14-53-011635448/log.log
('model_2_dev_mixed_average_0.6_topic_level.conll', 67.60893480430377)
```


Set this in [configs/gold_mixed_clustering1_0.40_test.json](configs/gold_mixed_clustering1_0.40_test.json)
```
  "model_num": 2,
  "threshold": 0.6
```


Running test:
```bash
# slurm
sh bin/launch_predict.sh configs/gold_mixed_clustering1_0.40_test.json $partition 32000
# single machine
sh bin/run_predict.sh configs/gold_mixed_clustering1_0.40_test.json $partition 32000
```

```
./scorer.pl  all ../data/ecb/gold/test_mixed_topic_level.conll ../models/pairwise_scorer1/topic_level_gold_mentions/mixed_0.40/test_mixed_average_0.6_model_2_topic_level.conll > ../models/pairwise_scorer1/topic_level_gold_mentions/mixed_0.40/test_mixed_average_0.6_model_2_topic_level.conll.score
```

```
grep -i -e metric -e Coref ../models/pairwise_scorer1/topic_level_gold_mentions/mixed_0.40/test_mixed_average_0.6_model_2_topic_level.conll.score
version: 8.01 /mnt/nfs/scratch1/nmonath/coref_public/reference-coreference-scorers/lib/CorScorer.pm
METRIC muc:
Coreference: Recall: (1972 / 2420) 81.48%       Precision: (1972 / 2553) 77.24% F1: 79.3%
METRIC bcub:
Coreference: Recall: (1795.33756998913 / 2800) 64.11%   Precision: (1666.7282545082 / 3022) 55.15%      F1: 59.29%
METRIC ceafm:
Coreference: Recall: (1719 / 2800) 61.39%       Precision: (1719 / 3022) 56.88% F1: 59.05%
METRIC ceafe:
Coreference: Recall: (207.363637842772 / 380) 54.56%    Precision: (207.363637842772 / 469) 44.21%      F1: 48.84%
METRIC blanc:
Coreference:
Coreference links: Recall: (15236 / 24784) 61.47%       Precision: (15236 / 30148) 50.53%       F1: 55.47%
Non-coreference links: Recall: (324014 / 393377) 82.36% Precision: (324014 / 460907) 70.29%     F1: 75.85%
```

## Predicted Mentions & Gold Topics (v2)

v2 Keeps `"max_mention_span": 10,` for the span_scorer step as in the original config file.
The remaining steps use the suggested `max_mention_span` value of 20.

Span scorer:

```bash
# slurm
sh bin/launch_span_scorer.sh configs/mixed_span_scorer2_0.40.json $partition 32000
# single machine
sh bin/run_span_scorer.sh configs/mixed_span_scorer2_0.40.json
```

Pairwise:

```bash
# slurm
sh bin/launch_pairwise.sh configs/mixed_pairwise1_0.40.json $partition 32000
# single machine
sh bin/launch_pairwise.sh configs/mixed_pairwise1_0.40.json
```

Tuning threshold:
```
# slurm
sh bin/launch_tuned_threshold.sh configs/mixed_clustering2_0.40.json $partition 32000
# single machine
sh bin/run_tuned_threshold.sh configs/mixed_clustering2_0.40.json
```

```bash
# slurm
sh bin/launch_find_best_model.sh models/pairwise_scorer2/topic_level_predicted_mentions/mixed_0.40 mixed $partition 32000
# single machine
sh bin/run_find_best_model.sh models/pairwise_scorer2/topic_level_predicted_mentions/mixed_0.40 mixed
```

```bash
tail -1 logs/run_scorer/mixed_0.40/2021-03-07-16-51-09-190845411/log.log
('model_3_dev_mixed_average_0.5_topic_level.conll', 45.94354329666605)
```

Set this in [configs/mixed_clustering2_0.40_test.json](configs/mixed_clustering2_0.40_test.json)
```
  "model_num": 3,
  "threshold": 0.5
```


Running test:
```bash
# slurm
sh bin/launch_predict.sh configs/mixed_clustering2_0.40_test.json $partition 32000
# single machine
sh bin/run_predict.sh configs/mixed_clustering2_0.40_test.json $partition 32000
```

Score test:
```
./scorer.pl  all ../data/ecb/gold/test_mixed_topic_level.conll ../models/pairwise_scorer2/topic_level_predicted_mentions/mixed_0.40/test_mixed_average_0.5_model_5_topic_level.conll > ../models/pairwise_scorer2/topic_level_predicted_mentions/mixed_0.40/test_mixed_average_0.5_model_5_topic_level.conll.score
```

```bash
