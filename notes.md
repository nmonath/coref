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

Set this in [configs/mixed_clustering1_0.40_test.json]
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
```

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