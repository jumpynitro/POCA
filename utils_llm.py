import random
import re
import sys

import numpy as np
import pandas as pd

# Constants
N_MIN_SAMPLE = 2
N_MIN = 0.25
N_MAX = 0.5
N_TOTAL_DATA = 10000


def obtain_str_mistral_dict(obs_value_dict, unobs_value_dict, recover=False):
    text = (
        f"<s>[INST] Complete the missing features tagged as [MASK] from this dictionary: "
        f"{obs_value_dict}? [/INST]"
    )
    if not recover:
        text += f"{unobs_value_dict} </s>"
    return text


def obtain_str_decilm_dict(obs_value_dict, unobs_value_dict, recover=False):
    text = (
        f"<s>Complete the missing features tagged as [MASK] from this dictionary: "
        f"{obs_value_dict}? ###Response: "
    )
    if not recover:
        text += f"{unobs_value_dict}</s>"
    return text


def obtain_str_llama_dict(obs_value_dict, unobs_value_dict, recover=False):
    instruction = "Complete the missing features tagged as [MASK] from the dictionary provided"
    text = (
        f"<|start_header_id|>system<|end_header_id|> {instruction}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|> {obs_value_dict} <|eot_id|>"
    )
    if not recover:
        text += (
            f"<|start_header_id|>assistant<|end_header_id|> ####{unobs_value_dict} ####<|eot_id|>"
        )
    return text


def obtain_str_general_dict(
    obs_value_dict, unobs_value_dict, recover=False, eos_token=None
):
    instruction = (
        "Impute missing features tagged as [MASK] using the provided dictionary as a context."
    )
    prompt_used = (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
    )
    prompt_used += f"### Instruction:\n{instruction}\n\n"
    prompt_used += f"### Input:\n{obs_value_dict}\n\n"
    if not recover:
        prompt_used += f"### Response:\n{unobs_value_dict}{eos_token}"
        return {"text": prompt_used}
    else:
        prompt_used += "### Response:\n"
        return prompt_used


def obtain_str_mistral(obs_value_str, unobs_value_str, unobs_str, recover=False):
    text = (
        f"<s>[INST] {obs_value_str}. What are the values of {unobs_str}? [/INST]"
    )
    if not recover:
        text += f"{unobs_value_str} </s>"
    return text


def obtain_str_decilm(obs_value_str, unobs_value_str, unobs_str, recover=False):
    text = (
        f"{obs_value_str}. What are the values of {unobs_str}? ###Response: "
    )
    if not recover:
        text += unobs_value_str
    return text


def obtain_str_llama(obs_value_str, unobs_value_str, unobs_str, recover=False):
    instruction = "Given the observed features, predict the features in question."
    text = (
        f"<|start_header_id|>system<|end_header_id|> {instruction}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|> {obs_value_str}. What are the values of "
        f"{unobs_str}? <|eot_id|>"
    )
    if not recover:
        text += (
            f"<|start_header_id|>assistant<|end_header_id|> ####{unobs_value_str}####<|eot_id|>"
        )
    return text


def formatting_func(
    row_data,
    all_columns,
    is_train=True,
    recover=False,
    use_random=False,
    is_syn_data=False,
    missing_with_relevance=False,
    is_mistral=True,
    llm_used=None,
    prompt_dict=None,
    eos_token=None,
):
    observed_list = [
        name for name in all_columns if row_data[f"mask_{name}"] is True
    ]

    if is_train:
        if not is_syn_data:
            if not missing_with_relevance:
                unobserved_list = observed_list
                len_data_used = len(observed_list)
                len_observed = int(random.uniform(N_MIN, N_MAX) * len_data_used)
                len_observed = max(N_MIN_SAMPLE, len_observed)
                shuffle_idx = list(range(len(observed_list)))
                random.shuffle(shuffle_idx)
                idx_obs = shuffle_idx[:len_observed]
                idx_unobs = shuffle_idx[len_observed:]
            else:
                unobserved_list = observed_list
                len_data_used = len(observed_list)
                len_observed = random.randint(N_MIN_SAMPLE, len_data_used - 1)
                total_idx = list(range(len(observed_list)))
                idx_obs = total_idx[:len_observed]
                idx_unobs = total_idx[len_observed:]
        else:
            unobserved_list = observed_list
            total_idx = list(range(len(observed_list)))
            idx_obs = total_idx[:1]
            idx_unobs = total_idx[1:]
    else:
        idx_obs = list(range(len(observed_list)))
        unobserved_list = [
            name for name in all_columns if row_data[f"mask_{name}"] is False
        ]
        idx_unobs = list(range(len(unobserved_list)))
        if use_random and not is_syn_data and not missing_with_relevance:
            random.shuffle(idx_obs)
            random.shuffle(idx_unobs)

    if prompt_dict:
        obs_value_str = ", ".join(
            [f"{observed_list[i]}: {row_data[observed_list[i]]}" for i in idx_obs]
        )
        unobs_mask_str = ", ".join(
            [f"{unobserved_list[i]}: [MASK]" for i in idx_unobs]
        )
        unobs_value_str = ", ".join(
            [f"{unobserved_list[i]}: {row_data[unobserved_list[i]]}" for i in idx_unobs]
        )

        instruct_str = f"{{{obs_value_str}, {unobs_mask_str}}}"
        response_str = f"{{{obs_value_str}, {unobs_value_str}}}"

        if llm_used == "mistral":
            return obtain_str_mistral_dict(instruct_str, response_str, recover)
        elif llm_used == "decilm":
            return obtain_str_decilm_dict(instruct_str, response_str, recover)
        elif llm_used == "llama":
            return obtain_str_llama_dict(instruct_str, response_str, recover)
        elif llm_used in [
            "llama3-unsloth",
            "phi3-unsloth",
            "mistral3-unsloth",
            "llama3.1-unsloth",
            "gemma2-unsloth",
        ]:
            return obtain_str_general_dict(
                instruct_str, response_str, recover, eos_token
            )
    else:
        obs_value_str = ", ".join(
            [f"{observed_list[i]} is {row_data[observed_list[i]]}" for i in idx_obs]
        )
        unobs_value_str = ", ".join(
            [f"{row_data[unobserved_list[i]]}" for i in idx_unobs]
        )
        unobs_str = ", ".join([f"{unobserved_list[i]}" for i in idx_unobs])

        if llm_used == "mistral":
            return obtain_str_mistral(
                obs_value_str, unobs_value_str, unobs_str, recover
            )
        elif llm_used == "decilm":
            return obtain_str_decilm(
                obs_value_str, unobs_value_str, unobs_str, recover
            )
        elif llm_used == "llama":
            return obtain_str_llama(
                obs_value_str, unobs_value_str, unobs_str, recover
            )


def obtain_new_data(data_used, new_n_rows):
    n_rows = len(data_used)
    if new_n_rows > n_rows:
        repeat_times = new_n_rows // n_rows
        new_df = pd.concat([data_used] * repeat_times, ignore_index=True)
        remaining_rows = new_n_rows % n_rows
        new_df = pd.concat(
            [new_df, data_used.head(remaining_rows)], ignore_index=True
        )
    else:
        new_df = data_used[:new_n_rows]
    return new_df


def update_table(data_used, mask_used):
    mask_used.columns = [f"mask_{col}" for col in mask_used.columns]
    return pd.concat([data_used, mask_used], axis=1)


def obtain_dict_of_values(df, categorical_columns):
    result_dict = {}
    for column in df.columns:
        if categorical_columns is None or column not in categorical_columns:
            result_dict[column] = "is_numerical"
        else:
            result_dict[column] = df[column].unique().tolist()
    return result_dict


def get_historical_and_pool_used(data):
    pool_set_type = data.main_dataset.pool_set_type
    (
        data_pool_used,
        index_pool_used,
        mask_pool_used,
    ) = data.main_dataset.get_original_all_pool_data()
    (
        data_historical_used,
        index_historical_used,
        mask_historical_used,
    ) = data.main_dataset.get_original_historical_data()

    data_final_pool_used = update_table(data_pool_used, mask_pool_used)
    data_final_historical_used = update_table(
        data_historical_used, mask_historical_used
    )
    all_columns = data_pool_used.columns

    print("len all pool data: ", len(data_final_pool_used))
    print("len all historical data: ", len(data_final_historical_used))

    if pool_set_type == "Pool":
        final_data = obtain_new_data(data_final_pool_used, N_TOTAL_DATA)
    elif pool_set_type in ["All", "Hist"]:
        final_data = obtain_new_data(data_final_historical_used, N_TOTAL_DATA)
    elif pool_set_type == "Hist-Pool":
        final_data_pool = obtain_new_data(
            data_final_pool_used, N_TOTAL_DATA // 2
        )
        final_data_historical = obtain_new_data(
            data_final_historical_used, N_TOTAL_DATA // 2
        )
        final_data = pd.concat(
            [final_data_pool, final_data_historical], ignore_index=True
        )
    return final_data, all_columns

def get_pool_hist_data(data):
    (
        data_pool_used,
        _,
        mask_pool_used,
    ) = data.main_dataset.get_original_all_pool_data()
    (
        data_historical_used,
        _,
        mask_historical_used,
    ) = data.main_dataset.get_original_historical_data()
    return data_pool_used, mask_pool_used, data_historical_used, mask_historical_used

def get_train_pool_vanilla(data):
    from utils_llm import update_table, obtain_dict_of_values

    pool_set_type = data.main_dataset.pool_set_type
    (
        data_pool_used,
        index_pool_used,
        mask_pool_used,
    ) = data.main_dataset.get_original_all_pool_data()
    (
        data_historical_used,
        index_historical_used,
        mask_historical_used,
    ) = data.main_dataset.get_original_historical_data()

    historical_data = data_historical_used.where(
        mask_historical_used, np.nan
    )
    pool_data = data_pool_used.where(mask_pool_used, np.nan)
    return historical_data, pool_data


def extract_value_from_string_dict(cfg, string, key):
    llm_name = cfg.llm_cfg.llm_name

    if llm_name in ["mistral", "decilm"]:
        response_marker = "[/INST]" if llm_name == "mistral" else "###Response:"
        response_start = string.find(response_marker)
        if response_start == -1:
            return None
        response_section = string[response_start:]
    elif llm_name == "llama":
        matches = re.findall(r"###(.*?)###", string)
        try:
            response_section = matches[0]
        except IndexError:
            return None
    else:
        response_marker = "### Response:"
        response_start = string.find(response_marker)
        if response_start == -1:
            return None
        response_section = string[response_start:]

    pattern = f", {key}:"
    start_index = response_section.find(pattern)
    if start_index == -1:
        return None
    start_index += len(pattern)
    end_index = response_section.find(",", start_index)
    if end_index == -1:
        end_index = response_section.find("}", start_index)
        if end_index == -1:
            return None
    final_string_used = response_section[start_index:end_index].lstrip()
    value = final_string_used.strip()
    return value


def extract_final_values_dict(cfg, text_example, result_dict, key_list):
    final_dict = {}
    for key in key_list:
        this_result_dict = result_dict[key]
        try:
            extracted_value = extract_value_from_string_dict(cfg, text_example, key)
            if extracted_value is None:
                return None
        except Exception:
            return None

        if this_result_dict == "is_numerical":
            try:
                final_dict[key] = float(extracted_value)
            except ValueError:
                return None
        else:
            value_str = str(extracted_value).strip('"')
            if value_str in this_result_dict:
                final_dict[key] = value_str
            else:
                return None
    return final_dict


def obtain_these_names_dict(
    cfg,
    row_dict,
    text,
    is_print=False,
    log_path=None,
    all_columns=None,
    dict_of_all=None,
):
    observed_list = [
        name for name in all_columns if row_dict[f"mask_{name}"] is True
    ]
    new_dict_f = {name: row_dict[name] for name in observed_list}
    new_dict   = new_dict_f.copy()
    unobserved_list = [
        name for name in all_columns if row_dict[f"mask_{name}"] is False
    ]
    is_valid = True

    if unobserved_list:
        dict_final = extract_final_values_dict(
            cfg, text, dict_of_all, unobserved_list
        )
        if dict_final is None:
            is_valid = False
            is_print = True
        else:
            new_dict_f.update(dict_final)
            new_dict = {col: new_dict_f[col] for col in all_columns}

    if is_print and log_path:
        original_data = {
            name: row_dict[name] for name in observed_list + unobserved_list
        }
        original_stdout = sys.stdout
        with open(log_path, "a") as log_file:
            sys.stdout = log_file
            if not is_valid:
                print("#" * 25 + " FAILED " + "#" * 25)
            print("#" * 56)
            print("#################### Original ######################")
            print(original_data)
            if is_valid:
                print("#################### Generated ######################")
                print(new_dict)
            print("#################### Text Gen ####################")
            print(text)
        sys.stdout = original_stdout

    if not is_valid:
        return None
    return new_dict


def remove_one_occurrence(input_list, value):
    found = False
    result = []
    for element in input_list:
        if element == value and not found:
            found = True
        else:
            result.append(element)
    return result


def extract_string(input_str):
    pattern1 = r"\[\[([^]]*)\]\]"
    pattern2 = r"\[([^]]*)\]"

    match1 = re.search(pattern1, input_str)
    match2 = re.search(pattern2, input_str)

    if match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    else:
        return input_str


def extract_numeric_value(input_string):
    pattern = r"[-+]?\d*\.?\d+|\d+"
    match = re.search(pattern, input_string)
    if match:
        return float(match.group())
    else:
        return None


def extract_final_values(cfg, text_example, result_dict):
    llm_name = cfg.llm_cfg.llm_name

    if llm_name == "mistral":
        pattern = re.compile(r"What are the values of (.+?)\? \[/INST\]")
    elif llm_name == "decilm":
        pattern = re.compile(r"What are the values of (.+?)\? ###Response: ")
    elif llm_name == "llama":
        pattern = re.compile(r"What are the values of (.+?)\?")

    match = pattern.search(text_example)
    if match:
        values_string = match.group(1)
        key_list = [value.strip() for value in values_string.split(",")]
    else:
        return None

    if llm_name == "mistral":
        inst_match = re.search(r"\[\/INST\](.*)", text_example, re.DOTALL)
    elif llm_name == "decilm":
        inst_match = re.search(r"###Response: (.*)", text_example, re.DOTALL)
    elif llm_name == "llama":
        matches = re.findall(r"###(.*?)###", text_example)

    try:
        if llm_name in ["mistral", "decilm"]:
            extracted_text = inst_match.group(1).replace("[/INST]", "").replace("</s>", "")
            list_of_values = extract_string(extracted_text).replace("[", "").replace("]", "").strip().split(", ")
        elif llm_name == "llama":
            list_of_values = matches[0].strip().split(", ")
    except (ValueError, IndexError, AttributeError):
        return None

    final_dict = {}
    is_first = True
    for idx, key in enumerate(key_list):
        this_result_dict = result_dict.get(key)
        if this_result_dict is None:
            return None

        try:
            value = list_of_values[idx]
        except IndexError:
            return None

        if this_result_dict == "is_numerical":
            try:
                if is_first:
                    final_dict[key] = extract_numeric_value(value)
                    is_first = False
                else:
                    final_dict[key] = float(value)
            except (ValueError, TypeError):
                return None
        else:
            value_str = value.strip('"')
            if is_first:
                is_found = any(name in value_str for name in this_result_dict)
                if is_found:
                    final_dict[key] = next(
                        name for name in this_result_dict if name in value_str
                    )
                else:
                    return None
                is_first = False
            elif value_str in this_result_dict:
                final_dict[key] = value_str
            else:
                return None
    return final_dict


def obtain_these_names_resp(
    cfg,
    row_dict,
    text,
    is_print=False,
    log_path=None,
    all_columns=None,
    dict_of_all=None,
):
    observed_list = [
        name for name in all_columns if row_dict[f"mask_{name}"] is True
    ]
    new_dict = {name: row_dict[name] for name in observed_list}
    unobserved_list = [
        name for name in all_columns if row_dict[f"mask_{name}"] is False
    ]
    is_valid = True

    if unobserved_list:
        dict_final = extract_final_values(cfg, text, dict_of_all)
        if dict_final is None:
            is_valid = False
            is_print = True
        else:
            new_dict.update(dict_final)

    if is_print and log_path:
        original_data = {
            name: row_dict[name] for name in observed_list + unobserved_list
        }
        original_stdout = sys.stdout
        with open(log_path, "a") as log_file:
            sys.stdout = log_file
            if not is_valid:
                print("#" * 25 + " FAILED " + "#" * 25)
            print("#" * 56)
            print("#################### Original ######################")
            print(original_data)
            if is_valid:
                print("#################### Generated ######################")
                print(new_dict)
            print("#################### Text Gen ####################")
            print(text)
        sys.stdout = original_stdout

    if not is_valid:
        return None
    return new_dict
