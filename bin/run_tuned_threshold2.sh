#!/usr/bin/env bash

set -exu

config=$1
threads=$2


export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

python tuned_threshold2.py --config $config