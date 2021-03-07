#!/usr/bin/env bash

set -exu

file_dir=$1
mention_type=$2
threads=$3

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

python run_scorer.py $file_dir $mention_type remove_singletons