#!/usr/bin/env zsh

set -exu

config=$1
partition=$2
mem=${3:-10000}
threads=${4:-8}

TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

model_name="pairwise"
dataset=`basename $config`
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/log

mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --gres=gpu:1 \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-4:00 \
            bin/run_predict.sh $config $threads
