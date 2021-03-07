#!/usr/bin/env zsh

set -exu

filedir=$1
mentiontype=$2
partition=$3
mem=${4:-10000}
threads=${5:-8}

TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

model_name="run_scorer"
dataset=`basename $filedir`
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
            bin/run_find_best_model.sh $filedir $mentiontype $threads
