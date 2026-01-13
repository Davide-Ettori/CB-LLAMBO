# Script to run LLAMBO on all HPOBench tasks.


#!/bin/bash
trap "kill -- -$BASHPID" EXIT

if [ "${OPENAI_API_TYPE}" = "azure" ]; then
    ENGINE="gpt35turbo_20230727"
else
    ENGINE="gpt-3.5-turbo"
fi

for dataset in "australian" "blood_transfusion" "car" "credit_g" "kc1" "phoneme" "segment" "vehicle"
do
    for model in "rf" "xgb" "nn"
    do
        echo "dataset: $dataset, model: $model"
        python3 exp_hpo_bench/run_hpo_bench.py --dataset $dataset --model $model --seed 0 --num_seeds 1 --engine $ENGINE --sm_mode discriminative --sm_eval_mode mc
    done
done
