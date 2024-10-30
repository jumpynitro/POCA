#!/bin/bash
# Export environment variables
export TRANSFORMERS_CACHE="$path_mnt/huggingface"
export HF_HOME="$path_mnt/huggingface"
export CUDA_VISIBLE_DEVICES=0

this_seed=1
batch_size=400
for this_llm in 'mistral3-unsloth'
do
    for this_data in 'magic' 'cardio' 'banking' 'adult' 'housing'
    do
        add_name="-seed$this_seed-pdict-true"
        llm_dir="$path_mnt/new_LLM_Pool"
        # Run training script
        python unsloth_train_llm.py \
            data="uci/$this_data" \
            llm_cfg.pool_set_type='Pool' \
            llm_cfg.pool_set_n_samples=1000 \
            llm_cfg.set_feat_percentage=0.5 \
            llm_cfg.llm_name=$this_llm \
            llm_cfg.add_name=$add_name \
            llm_cfg.prompt_dict=true \
            llm_cfg.llm_dir="$llm_dir" \
            llm_cfg.llm_seed=$this_seed

        # Run evaluation script
        python unsloth_eval_llm.py \
            data="uci/$this_data" \
            llm_cfg.pool_set_type='Pool' \
            llm_cfg.pool_set_n_samples=1000 \
            llm_cfg.set_feat_percentage=0.5 \
            llm_cfg.llm_name=$this_llm \
            llm_cfg.add_name=$add_name \
            llm_cfg.prompt_dict=true \
            llm_cfg.batch_size_generation=$batch_size \
            llm_cfg.mc_samples=8 \
            llm_cfg.llm_dir="$llm_dir"
    done
done
