#!/bin/bash
# Set environment variables (Assuming path_mnt is already exported)
export TRANSFORMERS_CACHE="${path_mnt}/huggingface"
export HF_HOME="${path_mnt}/huggingface"

# Fixed parameters
partition_set="Hist"
llm_name="mistral3-unsloth"
seed=1
add_name="-seed${seed}-pdict-true"
llm_dir="${path_mnt}/new_LLM_${partition_set}"
results_dir="rf_results_${partition_set}-camera"

# Define datasets and objectives
datasets=("magic" "cardio" "banking" "adult" "housing")
objectives=("epig" "marginal_entropy" "meanstd")

# Loop over datasets and objectives
for data in "${datasets[@]}"; do
    for objective in "${objectives[@]}"; do
        acquisition_objective="${objective},full-${objective},${objective}-po"
        python main.py --multirun "rng.seed=range(60)" \
            data="uci/${data}" \
            results_main_dir="${results_dir}" \
            acquisition.objective="${acquisition_objective}" \
            model=random_forest \
            trainer=random_forest \
            llm_cfg.llm_name="${llm_name}" \
            llm_cfg.pool_set_type="${partition_set}" \
            llm_cfg.pool_set_n_samples=1000 \
            llm_cfg.set_feat_percentage=0.5 \
            feature_selection.imputation_type=LLM \
            llm_cfg.add_name="${add_name}" \
            llm_cfg.prompt_dict=true \
            llm_cfg.llm_dir="${llm_dir}"
    done
done
