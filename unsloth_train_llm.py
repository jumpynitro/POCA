import os
import sys
import torch
import wandb
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import call, instantiate
from datasets import load_dataset
from unsloth import FastLanguageModel
from utils_llm import get_historical_and_pool_used, formatting_func
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Configuration parameters
max_seq_length = 2048  # Choose any value. Auto-support for RoPE scaling internally.
dtype = None  # Auto detection: use Float16 for Tesla T4/V100, Bfloat16 for Ampere+.
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Set to False if not needed.

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    # Initialize random number generator and dataset
    rng = call(cfg.rng)
    data = instantiate(cfg.data, rng=rng, seed=cfg.rng.seed, partially_observed=cfg.partially_observed, llm_cfg=cfg.llm_cfg)

    # Unpack LLM configuration parameters
    MAX_STEPS = cfg.llm_cfg.max_steps
    LR_RATE = cfg.llm_cfg.lr_rate
    CFG_R = cfg.llm_cfg.cfg_r
    CFG_LORA_ALPHA = cfg.llm_cfg.cfg_lora_alpha
    PER_DEVICE_BATCH = cfg.llm_cfg.per_device_batch
    MISSING_WITH_RELEVANCE = cfg.llm_cfg.missing_with_relevance
    IS_SYN_DATA = 'syndata' in data.main_dataset.this_dataset_name

    ############################## LOAD DATA ##################################
    CHECKPOINT_PATH_LLM = data.main_dataset.get_dir_llm_checkpoint()
    OUTPUT_PATH_LLM = data.main_dataset.get_dir_llm_outputs()
    
    path_to_check = f"{CHECKPOINT_PATH_LLM}/final-checkpoint-instruct/adapter_config.json"
    if os.path.exists(path_to_check):
        print(f"The path {path_to_check} exists. Exiting the script.")
        sys.exit()
    else:
        print(f"The path {path_to_check} does not exist. Continue with the script.")
    
    # Get historical data and save to JSONL format
    final_data, all_columns = get_historical_and_pool_used(data)
    data_train_path, _      = data.main_dataset.get_dataset_historical_path_jsonl()
    final_data.to_json(data_train_path, orient='records', lines=True)

    ############################ LOAD BASE MODEL ##############################
    # Base model selection based on LLM configuration
    model_id_map = {
        'mistral': "mistralai/Mistral-7B-Instruct-v0.2",
        'decilm': "Deci/DeciLM-7B-instruct",
        'llama': "meta-llama/Meta-Llama-3-8B-Instruct",
        'llama3-unsloth': "unsloth/llama-3-8b-Instruct-bnb-4bit",
        'phi3-unsloth': "unsloth/Phi-3-medium-4k-instruct",
        'mistral3-unsloth': "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        'llama3.1-unsloth': "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        'gemma2-unsloth': "unsloth/gemma-2-9b-bnb-4bit"
    }

    base_model_id = model_id_map.get(cfg.llm_cfg.llm_name)

    # Load the base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    ############################# LoRA ########################################
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Suggested values: 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Optimized with 0
        bias="none",  # Optimized with "none"
        use_gradient_checkpointing="unsloth",  # Optimized for long context with "unsloth"
        random_state=3407,
        use_rslora=False,  # Support for rank-stabilized LoRA
        loftq_config=None  # Option for LoftQ
    )

    ############################# Formatting ##################################
    # Data formatting function
    formatting_func_train = lambda row_data: formatting_func(
        row_data, all_columns, is_syn_data=IS_SYN_DATA,
        missing_with_relevance=MISSING_WITH_RELEVANCE, is_train=True,
        recover=False, use_random=False, llm_used=cfg.llm_cfg.llm_name,
        prompt_dict=cfg.llm_cfg.prompt_dict, eos_token=tokenizer.eos_token
    )

    # Load and format training dataset
    train_dataset = load_dataset('json', data_files=data_train_path, split='train')
    train_dataset = train_dataset.map(formatting_func_train)

    ############################ Train ########################################
    # Setup training arguments and trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Speed up training for short sequences
        args=TrainingArguments(
            output_dir=OUTPUT_PATH_LLM,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=10000,
            learning_rate=1e-4,  # LR_RATE and MAX_STEPS are available if needed
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            weight_decay=0.001,
            warmup_steps=100,
            logging_steps=500,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=42 + cfg.llm_cfg.llm_seed,
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=100000,
            evaluation_strategy="no",
            do_eval=False,
            max_grad_norm=1.0,
            report_to="none"
        ),
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(f"{CHECKPOINT_PATH_LLM}/final-checkpoint-instruct")
    tokenizer.save_pretrained(f"{CHECKPOINT_PATH_LLM}/final-checkpoint-instruct")

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full error stack trace
    main()
