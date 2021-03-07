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

Running test:
```

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