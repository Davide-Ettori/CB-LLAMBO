# Script to evaluate candidate point sampler.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

if [ "${OPENAI_API_TYPE}" = "azure" ]; then
    ENGINE="gpt35turbo_20230727"
else
    ENGINE="gpt-3.5-turbo"
fi

for dataset in "breast" "wine" "digits" "diabetes"
do
    for model in "RandomForest"
    do
        for num_observed in 5 10 20 30
        do
            python3 exp_evaluate_sampling/evaluate_sampling.py --dataset $dataset --model $model --num_observed $num_observed --num_seeds 1 --engine $ENGINE
        done
    done
done
