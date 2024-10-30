#!/bin/bash
# Export environment variables
export TRANSFORMERS_CACHE="$path_mnt/huggingface"
export HF_HOME="$path_mnt/huggingface"
export CUDA_VISIBLE_DEVICES=0


for this_llm in 'llama3.1-unsloth' 'gemma2-unsloth' 'mistral3-unsloth'
do
    # Set batch size based on LLM name
    if [ "$this_llm" == "gemma2-unsloth" ]; then
        batch_size=300
    else
        batch_size=400  # Default batch size if needed
    fi

    for this_data in 'magic' 'cardio' 'banking' 'adult' 'housing'
    do
        # Modify this_seed based on conditions
        if [ "$this_llm" == "gemma2-unsloth" ] && { [ "$this_data" == "banking" ] || [ "$this_data" == "cardio" ]; }; then
            this_seed=2
        else
            this_seed=1
        fi
        add_name="-seed$this_seed-pdict-true"
        llm_dir="$path_mnt/new_LLM_Hist"
        # Run training script
        python unsloth_train_llm.py \
            data="uci/$this_data" \
            llm_cfg.pool_set_type='Hist' \
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
            llm_cfg.pool_set_type='Hist' \
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
