import argparse
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scienceplots
import json
import os

plt.style.use('science')

# Constants
N_RUNS = 150
N_TOTAL_SEEDS = 60
FILTER_RUNS = True
NAME_FILE = 'testing'
ALL_COST = None  # Define this if needed

def get_results_dir(root_folder, data, llm, pool_type, seed_g, pdict, objective, this_seed):
    llm_config = 'MAX_STEPS_10000_LR_7.5e-05_R_32_ALPHA_64_PB_2'
    add_name = f"-seed{seed_g}-pdict-{pdict}"
    if data != 'banking':
        config = f'feat_per_0.5_pool_n_samples_1000_pool_type_{pool_type}{add_name}'
    else:
        config = f'feat_per_0.5_pool_n_samples_400_pool_type_{pool_type}{add_name}'
    this_dir = f"{root_folder}/{data}/{llm}/{config}/{llm_config}/{objective}/seed{this_seed}"
    return this_dir

def obtain_data(root_folder, objective, data, llm, pool_type, pdict, seed_g, IS_COST):
    all_results = []
    for this_seed in range(N_TOTAL_SEEDS):
        results_dir = get_results_dir(root_folder, data, llm, pool_type, seed_g, pdict, objective, this_seed)
        try:
            if NAME_FILE == 'cost_log':
                data_readed = pd.read_csv(f'{results_dir}/{NAME_FILE}.csv').cost.to_numpy()
            else:
                if not IS_COST:
                    print("####", f'{results_dir}/{NAME_FILE}.csv')
                    data_readed = pd.read_csv(f'{results_dir}/{NAME_FILE}.csv').test_acc.to_numpy()
                else:
                    data_acc = pd.read_csv(f'{results_dir}/testing.csv').test_acc.to_numpy()
                    data_cost = pd.read_csv(f'{results_dir}/cost_log.csv').cost.to_numpy()
                    data_readed = np.maximum.accumulate(compute_corresponding_cost(data_acc, data_cost, ALL_COST))
        except Exception as e:
            print(f"Not seed: {this_seed}", "  ", results_dir, "Exception:", e)
            continue
        data_readed = data_readed[:N_RUNS]
        if FILTER_RUNS and not IS_COST:
            if len(data_readed) == N_RUNS:
                all_results.append(data_readed)
        else:
            all_results.append(data_readed)
    try:
        all_results_np = np.stack(all_results, 0)
    except Exception as e:
        print("Failed to stack results:", e)
        for lal in all_results:
            print(lal.shape)
        return np.zeros(N_RUNS), np.zeros(N_RUNS)
    print("Number of runs included:", len(all_results))
    this_std = all_results_np.std(0) * 1.96 / math.sqrt(N_TOTAL_SEEDS)
    return all_results_np.mean(0), this_std

def get_results_in_dict(all_dict_exp, root_folder, data, llm, pool_type, pdict, seed, IS_COST):
    print("Processing experiments:", all_dict_exp.keys())
    for key in all_dict_exp.keys():
        aux_results = obtain_data(root_folder, key, data, llm, pool_type, pdict, seed, IS_COST)
        all_dict_exp[key]['results_mean'] = aux_results[0]
        all_dict_exp[key]['results_std'] = aux_results[1]
    return all_dict_exp

def obtain_full_name(model):
    mapping = {
        'bald': 'bald_LLM_FO',
        'epig': 'epig_LLM_FO',
        'meanstd': 'meanstd_LLM_FO',
        'marginal_entropy': 'marginal_entropy_LLM_FO'
    }
    return mapping.get(model, model)

def make_plot2(all_dict_exp, figure_size=(8, 5), name_path='someplot_aux.pdf',
               max_cols=1, this_n_axis=None, max_rows=1, is_cost=False, from_n=0):
    mcolors = {
        'b': '#0C5DA5',
        'o': '#FF9500',
        'g': '#00B945',
        'r': '#FF2C00',
        'p': '#845B97',
        'teal': '#008080',
        'bteal': '#025043',
        'lightgray': '#d3d3d3',
        'pink': '#FFC0CB',
        'bpink': '#FF8DA1',
        'gray': '#808080',
        'ymustard': '#ff8b47',
        'o2': '#e57858',
        'es': '#0097ab'
    }

    ax = plt.subplot(max_rows, max_cols, this_n_axis + 1)
    for key in all_dict_exp.keys():
        results_mean = all_dict_exp[key]['results_mean'] * 100
        results_std = all_dict_exp[key]['results_std'] * 100
        ncolor = all_dict_exp[key]['color']
        nlabel = all_dict_exp[key]['name']
        ar_used = np.arange(len(results_mean))
        if is_cost:
            ar_used = ALL_COST
        if 'FO' not in key:
            plt.plot(ar_used[from_n:], results_mean[from_n:], label=nlabel, linewidth=2, alpha=0.8,
                     color=mcolors.get(ncolor, 'black'))
            plt.fill_between(ar_used[from_n:], results_mean[from_n:] - results_std[from_n:],
                             results_mean[from_n:] + results_std[from_n:], alpha=0.2,
                             color=mcolors.get(ncolor, 'black'))
        else:
            new_result = [results_mean[-1], results_mean[-1]]
            plt.plot([from_n, len(results_mean)], new_result, label=nlabel, linewidth=2.5, alpha=0.3,
                     color=mcolors.get(ncolor, 'black'), linestyle='--')

    plt.grid()
    plt.xlabel('Number of instances')
    if this_n_axis % max_cols == 0:
        plt.ylabel('Test accuracy (\%)')
    if from_n is not None:
        plt.xlim(left=from_n)

def main():
    parser = argparse.ArgumentParser(description='Generate plots for the paper')
    parser.add_argument('--root_folder', type=str, default=None,
                        help='Root folder for the results')
    parser.add_argument('--models_used', type=str, nargs='+', default=None,
                        help='List of models to use')
    parser.add_argument('--llm_used', type=str, nargs='+', default=None,
                        help='List of LLMs to use')
    parser.add_argument('--data_sets', type=str, nargs='+',
                        default=['magic', 'adult', 'banking', 'cardio', 'housing'],
                        help='List of datasets to use')
    parser.add_argument('--this_seed', type=int, default=1,
                        help='Seed value')
    parser.add_argument('--output_file', type=str, default='new-models-seed1.pdf',
                        help='Output file name')
    parser.add_argument('--from_n', type=int, default=50,
                        help='Starting point for x-axis')
    parser.add_argument('--is_cost', action='store_true',
                        help='Plot cost instead of accuracy')
    parser.add_argument('--experiment_type', type=str, default='vary_llm',
                        choices=['vary_llm', 'vary_model', 'pool_camera', 'vary_model_main', 'specific'],
                        help='Type of experiment to run')
    args = parser.parse_args()

    # Set variables based on arguments
    IS_COST = args.is_cost
    THIS_SEED = args.this_seed
    from_n = args.from_n
    change_model_name = {'bald': 'EIG', 'epig': 'EPIG', 'marginal_entropy': 'Marginal Entropy', 'meanstd': 'Mean STD'}
    change_llm_name   = {'mistralv.03': 'Mistral v0.3', 'llama3.1': 'Llama 3.1', 'gemma2': 'Gemma 2'}

    # Adjust settings based on experiment type
    if args.experiment_type == 'vary_model_main':
        root_folder = 'rf_results_Hist' if not args.root_folder else args.root_folder
        models_used = ['bald', 'epig'] if not args.models_used else args.models_used
        llm_used_list = ['mistral3-unsloth'] if not args.llm_used else args.llm_used
        max_rows = 2
        max_cols = 5
        plt.figure(figsize=(18, 5))
    elif args.experiment_type == 'vary_llm':
        root_folder = 'rf_results_Hist' if not args.root_folder else args.root_folder
        models_used = ['bald'] if not args.models_used else args.models_used
        llm_used_list = ['mistral3-unsloth', 'llama3.1-unsloth', 'gemma2-unsloth'] if not args.llm_used else args.llm_used
        max_rows = 3
        max_cols = 5
        plt.figure(figsize=(15, 12))
    elif args.experiment_type == 'vary_model':
        root_folder = 'rf_results_Hist' if not args.root_folder else args.root_folder
        models_used = ['bald', 'epig', 'marginal_entropy', 'meanstd'] if not args.models_used else args.models_used
        llm_used_list = ['mistral3-unsloth'] if not args.llm_used else args.llm_used
        max_rows = 4
        max_cols = 5
        plt.figure(figsize=(15, 16))
    elif args.experiment_type == 'pool_camera':
        root_folder = 'rf_results_Pool' if not args.root_folder else args.root_folder
        models_used = ['bald'] if not args.models_used else args.models_used
        llm_used_list = ['mistral3-unsloth'] if not args.llm_used else args.llm_used
        max_rows = 1
        max_cols = 5
        plt.figure(figsize=(15, 3))
    else:
        root_folder = args.root_folder if args.root_folder else 'rf_results_Hist'
        models_used = args.models_used if args.models_used else ['bald']
        llm_used_list = args.llm_used if args.llm_used else ['mistral3-unsloth']
        max_rows = 1
        max_cols = len(args.data_sets)
        plt.figure(figsize=(15/max_rows, 15/max_cols))

    # Prepare results_config_dict
    results_config_dict = {}
    for data_set in args.data_sets:
        if args.experiment_type == 'pool_camera':
            results_config_dict[data_set] = {'pool_type': 'Pool', 'pdict': 'true'}
        else:
            results_config_dict[data_set] = {'pool_type': 'Hist', 'pdict': 'true'}

    idx = 0
    for model in models_used:
        for llm_used in llm_used_list:
            for key in results_config_dict.keys():
                data, llm = key, llm_used
                pool_type, pdict = results_config_dict[key]['pool_type'], results_config_dict[key]['pdict']
                all_dict_exp = {}
                if args.experiment_type == 'vary_llm':
                    all_dict_exp['random_LLM'] = {'name': 'Random', 'color': 'gray'}
                    all_dict_exp[f'{model}_LLM'] = {'name': 'BALD', 'color': 'o2'}
                    all_dict_exp[f'{model}-po_LLM'] = {'name': '$\\mu$PO-EIG', 'color': 'teal'}
                    all_dict_exp[obtain_full_name(model)] = {'name': 'Oracle', 'color': 'teal'}
                elif args.experiment_type == 'vary_model' or args.experiment_type == 'vary_model_main' :
                    all_dict_exp['random_LLM'] = {'name': 'Random', 'color': 'gray'}
                    all_dict_exp[f'{model}_LLM'] = {'name': 'Vanilla AL metric', 'color': 'o2'}
                    all_dict_exp[f'{model}-po_LLM'] = {'name': '$\\mu$POCA metric (ours)', 'color': 'teal'}
                    all_dict_exp[obtain_full_name(model)] = {'name': 'Oracle', 'color': 'teal'}
                elif args.experiment_type == 'pool_camera':
                    all_dict_exp['random_LLM'] = {'name': 'Random', 'color': 'gray'}
                    all_dict_exp[f'{model}_LLM'] = {'name': 'Vanilla AL metric', 'color': 'o2'}
                    all_dict_exp[f'{model}-po_LLM'] = {'name': '$\\mu$POCA metric (ours)', 'color': 'teal'}
                    all_dict_exp[obtain_full_name(model)] = {'name': 'Oracle', 'color': 'teal'}
                else:
                    all_dict_exp['random_LLM'] = {'name': 'Random', 'color': 'gray'}
                    all_dict_exp[f'{model}_LLM'] = {'name': 'Vanilla AL metric', 'color': 'o2'}
                    all_dict_exp[f'{model}-po_LLM'] = {'name': '$\\mu$POCA metric (ours)', 'color': 'teal'}
                    all_dict_exp[obtain_full_name(model)] = {'name': 'Oracle', 'color': 'teal'}
                try:
                    seed_used =  2 if data in ['banking', 'cardio'] and llm_used == 'gemma2-unsloth' else THIS_SEED
                    get_results_in_dict(all_dict_exp, root_folder, data, llm, pool_type, pdict, seed_used, IS_COST)
                    make_plot2(all_dict_exp, max_rows=max_rows, max_cols=max_cols, this_n_axis=idx, from_n=from_n, is_cost=IS_COST)
                    name_llm = llm_used.replace('-unsloth', '')
                    name_llm = name_llm.replace('mistral3', 'mistralv.03')
                    if args.experiment_type == 'vary_llm':
                        plt.title(f'{change_llm_name[name_llm]}  --- {data}')
                    elif args.experiment_type == 'vary_model' or args.experiment_type == 'vary_model_main':
                        plt.title(f'{change_model_name[model]}  --- {data}')
                    else:
                        plt.title(f'{change_llm_name[name_llm]}  --- {data}')
                except Exception as e:
                    print(f"Failed for model {model}, data {data}, llm {llm}: {e}")
                idx += 1

    if args.experiment_type == 'vary_model_main' :
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.0, wspace=0.2, hspace=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(-1.8, -0.1), fontsize=12, ncol=4)

    else:
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15, wspace=0.4, hspace=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(-2.25, -0.1), fontsize=12, ncol=4)
    plt.savefig(args.output_file)
    print(f"Plot saved to {args.output_file}")

if __name__ == '__main__':
    main()


