# Set cache directories for Hugging Face
export TRANSFORMERS_CACHE="$path_mnt/huggingface"
export HF_HOME="$path_mnt/huggingface" 

# Loop through configurations and run main.py script
for this_llm in 'llama3.1-unsloth' 'gemma2-unsloth' 'mistral3-unsloth'; do
    for this_data in 'magic' 'cardio' 'banking' 'adult' 'housing'; do
        add_name="-seed$this_seed-pdict-true"
        python main.py --multirun 'rng.seed=range(60)' \
            data="uci/$this_data" \
            results_main_dir="rf_results_Pool-camera" \
            acquisition.objective=bald-po,bald,random,full-bald \
            model=random_forest \
            trainer=random_forest \
            llm_cfg.llm_name=$this_llm \
            llm_cfg.pool_set_type='Pool' \
            llm_cfg.pool_set_n_samples=1000 \
            llm_cfg.set_feat_percentage=0.5 \
            feature_selection.imputation_type=LLM \
            llm_cfg.add_name=$add_name \
            llm_cfg.prompt_dict=true \
            llm_cfg.llm_dir="$path_mnt/new_LLM_Pool"
    done
done
