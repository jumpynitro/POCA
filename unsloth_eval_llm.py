import os
import sys
import torch
import hydra
import pandas as pd
from omegaconf import DictConfig
from hydra.utils import call, instantiate
from datasets import load_dataset
from unsloth import FastLanguageModel

# Configuration constants
MAX_SEQ_LENGTH = 2048  # Choose any! RoPE scaling supported internally
LOAD_IN_4BIT = True    # Use 4-bit quantization to reduce memory usage, set to False if not needed
DTYPE = None           # Set to None for auto-detection, or specify (e.g., Float16 for Tesla T4, V100, Bfloat16 for Ampere+)

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    # RNG initialization
    rng = call(cfg.rng)
    
    # Data initialization
    data = instantiate(cfg.data, rng=rng, seed=cfg.rng.seed, partially_observed=cfg.partially_observed, llm_cfg=cfg.llm_cfg)
    
    # Constants from configuration
    MISSING_WITH_RELEVANCE = cfg.llm_cfg.missing_with_relevance
    IS_SYN_DATA = 'syndata' in data.main_dataset.this_dataset_name
    BATCH_SIZE_GENERATION = cfg.llm_cfg.batch_size_generation
    MC_SAMPLES = cfg.llm_cfg.mc_samples
    IS_TESTING = cfg.llm_cfg.is_testing

    # Paths
    CHECKPOINT_PATH_LLM = data.main_dataset.get_dir_llm_checkpoint()
    OUTPUT_PATH_LLM = data.main_dataset.get_dir_llm_outputs()

    # Check if output file exists
    path_to_check = f'{OUTPUT_PATH_LLM}/LLM_generated_mc_sample_{MC_SAMPLES-1}.txt'
    if os.path.exists(path_to_check):
        print(f"The path {path_to_check} exists. Exiting the script.")
        sys.exit()
    else:
        print(f"The path {path_to_check} does not exist. Continue with the script.")

    # Load utilities
    from utils_llm import update_table, obtain_dict_of_values
    data_pool_used, _, mask_pool_used = data.main_dataset.get_original_all_pool_data()
    data_final_pool_used = update_table(data_pool_used, mask_pool_used)
    all_columns = data_pool_used.columns
    categorical_columns = data.main_dataset.get_col_categorical()
    dict_of_all = obtain_dict_of_values(data_pool_used, categorical_columns)

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{CHECKPOINT_PATH_LLM}/final-checkpoint-instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    # Formatting function
    from utils_llm import formatting_func
    formatting_func_test_recover_random = lambda row_data: formatting_func(
        row_data, all_columns, is_syn_data=IS_SYN_DATA, missing_with_relevance=MISSING_WITH_RELEVANCE, 
        is_train=False, recover=True, use_random=True, llm_used=cfg.llm_cfg.llm_name, prompt_dict=cfg.llm_cfg.prompt_dict
    )

    # Evaluation function
    from utils_llm import obtain_these_names_dict, obtain_these_names_resp, remove_one_occurrence
    LOG_FILE_PATH = f'{OUTPUT_PATH_LLM}/LLM_generated.txt'
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)

    MAX_FAILURES = 20000
    index_to_process = list(data_pool_used.index)

    def process_data(index_to_process, is_print=False, log_path=None, print_every_n=10):
        all_dict = []
        all_index = []
        num_failed = 0

        while index_to_process:
            current_indexes = index_to_process[:BATCH_SIZE_GENERATION]
            list_of_dict = data_final_pool_used.loc[current_indexes].to_dict(orient='records')
            all_prompts = [formatting_func_test_recover_random(json_data) for json_data in list_of_dict]
            model_input = tokenizer(all_prompts, return_tensors="pt", padding=True).to("cuda")
            
            with torch.no_grad():
                decoded_list = tokenizer.batch_decode(model.generate(
                    **model_input, max_new_tokens=512, repetition_penalty=1.00,
                    do_sample=True, temperature=0.7, use_cache=True))

            for idx, this_index in enumerate(current_indexes):
                is_print_here = is_print and not len(index_to_process) % print_every_n
                
                if cfg.llm_cfg.prompt_dict:
                    dict_result = obtain_these_names_dict(cfg, list_of_dict[idx], decoded_list[idx], 
                                                          is_print=is_print_here, log_path=log_path,
                                                          all_columns=all_columns, dict_of_all=dict_of_all)
                else:
                    dict_result = obtain_these_names_resp(cfg, list_of_dict[idx], decoded_list[idx], 
                                                          is_print=is_print_here, log_path=log_path,
                                                          all_columns=all_columns, dict_of_all=dict_of_all)

                if dict_result:
                    index_to_process = remove_one_occurrence(index_to_process, this_index)
                    all_dict.append(dict_result)
                    all_index.append(this_index)

                    if is_print and not len(index_to_process) % 1000 and log_path:
                        with open(log_path, 'a') as log_file:
                            log_file.write(f"##################### {len(index_to_process)}  #######################\n")

                else:
                    num_failed += 1

                if num_failed == MAX_FAILURES:
                    print("Failed processing data. Exiting...")
                    sys.exit()

        final_pd = pd.DataFrame(all_dict)[all_columns]
        final_pd.index = all_index
        return final_pd

    print("Beginning processing...")
    
    # Helper function to count files
    def count_files(folder_path, prefix):
        files = os.listdir(folder_path)
        return len([file for file in files if file.startswith(prefix)])

    # Generate samples if not in testing mode
    SAMPLES_PATH = data.main_dataset.get_dir_llm_samples()
    if not IS_TESTING:
        for mc_sample in range(count_files(SAMPLES_PATH, 'mc_sample_'), MC_SAMPLES):
            LOG_FILE_PATH_THIS = f'{OUTPUT_PATH_LLM}/LLM_generated_mc_sample_{mc_sample}.txt'
            if os.path.exists(LOG_FILE_PATH_THIS):
                os.remove(LOG_FILE_PATH_THIS)

            final_pd = process_data(index_to_process, is_print=True, log_path=LOG_FILE_PATH_THIS, print_every_n=50)
            final_pd.to_csv(f'{SAMPLES_PATH}/mc_sample_{mc_sample}.csv')

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()
