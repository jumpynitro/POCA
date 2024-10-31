import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scienceplots

#plt.style.use('science')

# Constants
N_RUNS = 125
N_TOTAL_SEEDS = 60
FILTER_RUNS = True
NAME_FILE = 'testing'
ALL_COST = np.linspace(5, 75, 14)

def compute_corresponding_cost(All_acc, All_cost, restriction_cost):
    indices = np.searchsorted(All_cost, restriction_cost, side='right') - 1
    indices = np.clip(indices, 0, len(All_cost)-1)
    new_acc = np.maximum.accumulate(All_acc)
    return np.array(new_acc[indices])

def get_results_dir(data, llm, pool_type, seed_g, pdict, objective, this_seed, root_folder, use_neurips_seed=False):
    llm_config = 'MAX_STEPS_10000_LR_7.5e-05_R_32_ALPHA_64_PB_2'
    if use_neurips_seed:
        add_name = f"-neurips-seed{seed_g}-pdict-{pdict}"
    else:
        add_name = f"-seed{seed_g}-pdict-{pdict}"
    if data != 'banking':
        config = f'feat_per_0.5_pool_n_samples_1000_pool_type_{pool_type}{add_name}'
    else:
        config = f'feat_per_0.5_pool_n_samples_400_pool_type_{pool_type}{add_name}'
    this_dir = f"{root_folder}/{data}/{llm}/{config}/{llm_config}/{objective}/seed{this_seed}"
    return this_dir

def obtain_data(objective, data, llm, pool_type, pdict, seed_g, root_folder, is_cost=False):
    all_results = []
    for this_seed in range(N_TOTAL_SEEDS):
        results_dir = get_results_dir(data, llm, pool_type, seed_g, pdict, objective, this_seed, root_folder)
        try:
            if NAME_FILE == 'cost_log':
                data_readed = pd.read_csv(f'{results_dir}/{NAME_FILE}.csv').cost.to_numpy()
            else:
                if not is_cost:
                    data_readed = pd.read_csv(f'{results_dir}/{NAME_FILE}.csv').test_acc.to_numpy()
                else:
                    data_acc = pd.read_csv(f'{results_dir}/testing.csv').test_acc.to_numpy()
                    data_cost = pd.read_csv(f'{results_dir}/cost_log.csv').cost.to_numpy()
                    data_readed = np.maximum.accumulate(compute_corresponding_cost(data_acc, data_cost, ALL_COST))
            data_readed = data_readed[:N_RUNS]
            if FILTER_RUNS and not is_cost:
                if len(data_readed) == N_RUNS:
                    all_results.append(data_readed)
            else:
                all_results.append(data_readed)
        except Exception as e:
            print(f"Error processing seed {this_seed}: {e}")
            continue
    if len(all_results) == 0:
        raise ValueError("No data found for the given configuration.")
    all_results_np = np.stack(all_results, 0)
    this_std = all_results_np.std(0) * 1.96 / math.sqrt(N_TOTAL_SEEDS)
    return all_results_np.mean(0), this_std

def obtain_data_cost(objective, data, llm, pool_type, pdict, seed_g, root_folder, per_cost_feat_list, budget):
    all_results = []
    for this_seed in range(N_TOTAL_SEEDS):
        try:
            results_dir = get_results_dir(data, llm, pool_type, seed_g, pdict, objective, this_seed, root_folder)
            data_acc = pd.read_csv(f'{results_dir}/testing.csv').test_acc.to_numpy()
            data_cost = pd.read_csv(f'{results_dir}/cost_log.csv').cost.to_numpy()
        except:
            continue
        try:
            for per_cost_feat in per_cost_feat_list:
                new_cost = data_cost * per_cost_feat + (1 - per_cost_feat) * np.arange(1, len(data_cost) + 1)
                index = np.argmax(new_cost >= budget)
                if index == 0:
                    continue
                all_results.append({'acc': data_acc[index], 'per_feat': f'{int(per_cost_feat * 100)}\%', 'seed': this_seed})
        except Exception as e:
            print(f"Error processing seed {this_seed}: {e}")
            continue
    this_pd = pd.DataFrame(all_results)
    return this_pd

def make_plot_accuracy_vs_instances(all_dict_exp, ax, is_cost=False, from_n=0):
    mcolors = {
        'b':'#0C5DA5',
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
        #'ored': '#DD517F',
        'ored': '#845B97',
        'es': '#0097ab'
    }
    for key in all_dict_exp.keys():
        results_mean = all_dict_exp[key]['results_mean'] * 100
        results_std  = all_dict_exp[key]['results_std'] * 100
        ncolor       = all_dict_exp[key]['color']
        nlabel       = all_dict_exp[key]['name']
        ar_used = np.arange(len(results_mean))
        if is_cost:
            ar_used = ALL_COST
        ax.plot(ar_used[from_n:], results_mean[from_n:], label=nlabel, linewidth=2, alpha=0.8,
                color=mcolors[ncolor], zorder = 2)
        ax.fill_between(ar_used[from_n:], results_mean[from_n:] - results_std[from_n:], results_mean[from_n:] + results_std[from_n:], alpha=0.2,
                        color=mcolors[ncolor], zorder = 2)
    ax.grid()
    if not is_cost:
        ax.set_xlabel('Number of instances', fontsize=14)
    else:
        ax.set_xlabel('Budget spent in terms of features (no label cost considered)', fontsize=14)
    ax.set_ylabel('Test accuracy (\%)', fontsize = 14)
    # Remove legend from individual plots
    ax.legend().remove()

def make_plot_accuracy_bar(new_pd, ax):
    new_pd['acc_percent'] = new_pd['acc'] * 100
    #custom_palette = ['#808080', '#008080', '#e57858', '#DD517F'][::-1]

    custom_palette = ['#808080', '#008080', '#e57858', '#845B97'][::-1]

    sns.barplot(
        data=new_pd,
        hue='acq_feat',
        x='per_feat',
        y='acc_percent',
        palette=custom_palette,
        alpha=0.9,
        ax=ax,
        zorder=10  # Set a higher z-order for the bars
    )
    ax.set_xlabel('Percentage of the feature cost w.r.t. feature cost + label cost', fontsize=14)
    ax.set_ylabel('Test accuracy (\%)', fontsize = 14)
    # Remove legend from the subplot
    ax.legend().remove()
    ax.grid(axis='y', zorder = 5)  # Add gridlines along the y-axis

    # Adjust y-axis limits based on the bar heights and error bars
    # Collect all the bar heights and error bar values
    bar_heights = []
    error_mins = []
    error_maxs = []

    # Get the containers (bars) from the plot
    for container in ax.containers:
        # Each container corresponds to a group of bars (one per hue level)
        for bar in container:
            bar_heights.append(bar.get_height())

    # Get the error bars (lines)
    for i, line in enumerate(ax.lines):
        # Error bars are represented by lines; every 3 lines correspond to one bar:
        # - Line 0: center line of the error bar
        # - Line 1: lower cap
        # - Line 2: upper cap
        if i % 3 == 0:
            y_data = line.get_ydata()
            error_mins.append(y_data[0])
            error_maxs.append(y_data[1])

    # Combine all y-values
    all_y_values = bar_heights + error_mins + error_maxs

    # Find the min and max y-values
    min_y = min(all_y_values)
    max_y = max(all_y_values)

    # Add padding
    y_range = max_y - min_y
    padding = y_range * 0.3  # 10% padding

    # Ensure y-axis limits are within 0% to 100%
    lower_limit = max(0, min_y - padding)
    upper_limit = min(100, max_y + padding)

    # Set y-axis limits
    ax.set_ylim(lower_limit, upper_limit)

def main(args):
    # Prepare configurations
    results_config_dict = {
        'magic': {'pool_type': 'Hist', 'pdict': 'true'},
        'adult': {'pool_type': 'Pool', 'pdict': 'true'},
        'banking': {'pool_type': 'Pool', 'pdict': 'true'},
        'cardio': {'pool_type': 'Pool', 'pdict': 'true'},
        'housing': {'pool_type': 'Pool', 'pdict': 'true'},
    }
    model = args.model
    llm_used = args.llm
    data_key = args.data
    THIS_SEED = args.seed_g
    root_folder_base = f'rf_results_{results_config_dict[data_key]["pool_type"]}'
    pool_type = results_config_dict[data_key]['pool_type']
    pdict = results_config_dict[data_key]['pdict']
    data = data_key
    llm = llm_used
    seed_g = THIS_SEED
    # Prepare plot
    fig, axs = plt.subplots(1, 3, figsize=(22, 4.5))  # Increased figure width
    # Define objective names
    objective_names = {
        'bald-po-feature-0.2_LLM': '20\% feat. acquired (PO-EIG)',
        'bald-po-feature-0.6_LLM': '60\% feat. acquired (PO-EIG)',
        'bald-po_LLM': '100\% feat. acquired (PO-EIG)',
        'bald_LLM': '100\% feat. acquired (EIG)'
    }
    colors = {
        'bald_LLM': 'gray',
        'bald-po_LLM': 'teal',
        'bald-po-feature-0.2_LLM': 'ored',
        'bald-po-feature-0.6_LLM': 'o2'
    }
    # Prepare all_dict_exp
    all_dict_exp = {}
    for key in ['bald_LLM', 'bald-po_LLM', 'bald-po-feature-0.2_LLM', 'bald-po-feature-0.6_LLM']:
        all_dict_exp[key] = {
            'name': objective_names[key],
            'color': colors[key]
        }
    # PLOT2
    IS_COST = False
    for key in all_dict_exp.keys():
        aux_results = obtain_data(key, data, llm, pool_type, pdict, seed_g, root_folder_base, is_cost=IS_COST)
        all_dict_exp[key]['results_mean'] = aux_results[0]
        all_dict_exp[key]['results_std'] = aux_results[1]
    make_plot_accuracy_vs_instances(all_dict_exp, axs[0], is_cost=IS_COST)
    #axs[0].set_title('PLOT2')
    # PLOT1
    IS_COST = True
    for key in all_dict_exp.keys():
        aux_results = obtain_data(key, data, llm, pool_type, pdict, seed_g, root_folder_base, is_cost=IS_COST)
        all_dict_exp[key]['results_mean'] = aux_results[0]
        all_dict_exp[key]['results_std'] = aux_results[1]
    make_plot_accuracy_vs_instances(all_dict_exp, axs[1], is_cost=IS_COST)
    #axs[1].set_title('PLOT1')
    # PLOT3
    per_cost_feat_list = [0.5, 0.75, 1.0]
    budget = 50
    all_objectives = ['bald-po-feature-0.2_LLM', 'bald-po-feature-0.6_LLM', 'bald-po_LLM', 'bald_LLM']
    new_pd_list = []
    for obj in all_objectives:
        this_pd = obtain_data_cost(obj, data, llm, pool_type, pdict, seed_g, root_folder_base,
                                   per_cost_feat_list=per_cost_feat_list, budget=budget)
        this_pd['acq_feat'] = objective_names[obj]
        new_pd_list.append(this_pd)
    new_pd = pd.concat(new_pd_list, axis=0)
    make_plot_accuracy_bar(new_pd, axs[2])
    #axs[2].set_title('PLOT3')
    # Adjust layout and create shared legend
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=16, ncol=4, bbox_to_anchor=(0.545, -0.1))
    
    #plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust rect to make space for legend
    plt.savefig(args.output_file)

    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plots for active learning experiments.')
    parser.add_argument('--model', type=str, default='bald', help='Acquisition model to use.')
    parser.add_argument('--llm', type=str, default='mistral3-unsloth', help='LLM used in the experiments.')
    parser.add_argument('--data', type=str, default='magic', help='Dataset name.')
    parser.add_argument('--seed_g', type=int, default=1, help='Seed group.')
    parser.add_argument('--output_file', type=str, default='cost-model.pdf',
                        help='Output file name')
    args = parser.parse_args()
    main(args)

