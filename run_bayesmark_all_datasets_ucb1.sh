#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# Run all Bayesmark datasets and models using UCB1 surrogate mode.
if [ "${OPENAI_API_TYPE}" = "azure" ]; then
    ENGINE="gpt35turbo_20230727"
else
    ENGINE="gpt-3.5-turbo"
fi

for dataset in "digits" "wine" "diabetes" "iris" "breast"
do
    for model in "RandomForest" "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
    do
        python3 exp_bayesmark/run_bayesmark.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --sm_eval_mode bandit_ucb1 --engine $ENGINE
        sleep 60
    done
done
