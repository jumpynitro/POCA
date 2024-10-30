#!/bin/bash
# Export necessary environment variables
export TRANSFORMERS_CACHE="$path_mnt/huggingface"
export HF_HOME="$path_mnt/huggingface"

# Optional: Uncomment the following line to set CUDA devices
# export CUDA_VISIBLE_DEVICES=0

# Define parameters
partition_set="Hist"
results_dir="rf_results_${partition_set}-camera"
llm_dir="${path_mnt}/new_LLM_${partition_set}"
acquisition_objectives="bald-po-feature-0.2,bald-po-feature-0.6"

# Execute the Python script for the specified dataset
python main.py --multirun 'rng.seed=range(60)' \
    data="uci/magic" \
    results_main_dir="${results_dir}" \
    acquisition.objective="${acquisition_objectives}" \
    model=random_forest \
    trainer=random_forest \
    llm_cfg.llm_name="mistral3-unsloth" \
    llm_cfg.pool_set_type="${partition_set}" \
    llm_cfg.pool_set_n_samples=1000 \
    llm_cfg.set_feat_percentage=0.5 \
    feature_selection.imputation_type=LLM \
    llm_cfg.add_name="-seed1-pdict-true" \
    llm_cfg.prompt_dict=true \
    llm_cfg.llm_dir="${llm_dir}" \
    feature_selection.is_cost_based=true \
    feature_selection.cost_restriction=true \
    llm_cfg.max_budget=75
