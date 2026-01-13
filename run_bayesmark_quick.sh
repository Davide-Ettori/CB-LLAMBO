#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# Quick end-to-end pipeline smoke test (single dataset/model).
if [ "${OPENAI_API_TYPE}" = "azure" ]; then
    ENGINE="gpt35turbo_20230727"
else
    ENGINE="gpt-3.5-turbo"
fi

DATASET="breast"
MODEL="RandomForest"

python3 exp_bayesmark/run_bayesmark.py --dataset $DATASET --model $MODEL --num_seeds 1 --sm_mode discriminative --sm_eval_mode mc --engine $ENGINE
