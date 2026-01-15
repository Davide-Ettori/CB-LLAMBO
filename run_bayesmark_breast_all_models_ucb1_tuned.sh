#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# Run all Bayesmark models on breast dataset using UCB1-Tuned surrogate mode.
if [ "${OPENAI_API_TYPE}" = "azure" ]; then
    ENGINE="gpt35turbo_20230727"
else
    ENGINE="gpt-3.5-turbo"
fi

DATASET="diabetes"

for model in "RandomForest" "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
do
    python3 exp_bayesmark/run_bayesmark.py --dataset $DATASET --model $model --num_seeds 1 --sm_mode discriminative --sm_eval_mode bandit_ucb1_tuned --engine $ENGINE
    sleep 60
done
