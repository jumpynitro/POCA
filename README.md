## Partially Observable Cost-Aware Active-Learning with Large Language Models

Official code repository for NeurIPS'24 paper `Partially Observable Cost-Aware Active-Learning with Large Language Models`.

**Authors:** Nicolás Astorga, Tennison Liu, Nabeel Seedat, Mihaela van der Schaar

#### Abstract
Conducting experiments and collecting data for machine learning models is a complex and expensive endeavor, particularly when confronted with limited information. Typically, extensive *experiments* to obtain features and labels come with a significant acquisition cost, making it impractical to carry out all of them. Therefore, it becomes crucial to strategically determine what to acquire to maximize the predictive performance while minimizing costs. To perform this task, existing data acquisition methods assume the availability of an initial dataset that is both fully observed and labeled, crucially overlooking the *partial observability* of features characteristic of many real-world scenarios. In response to this challenge, we present Partially Observable Cost-Aware Active-Learning (POCA), a new learning approach aimed at improving model generalization in data-scarce and data-costly scenarios through label and/or feature acquisition. Introducing $\mu$POCA as an instantiation, we maximize the uncertainty reduction in the predictive model when obtaining labels and features, considering associated costs. $\mu$POCA enhances traditional Active Learning metrics based solely on the observed features by generating the unobserved features through Generative Surrogate Models, particularly Large Language Models (LLMs). We empirically validate $\mu$POCA across diverse tabular datasets, varying data availability, and acquisition costs.


---
# POCA Project

This repository provides code and instructions to run and reproduce experiments described in the POCA paper. The experiments focus on train LLM, generating mc samples, and train downstream models.

---

## 1. Setup

### Step 1: Set up the Conda environment

For GSM trainign and sampling we use unsloth training with the following configuration. At the moment of running the results torch (2.3 was installed, now torch 2.5 is automatically installed)
```
conda create --name poca python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate poca

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

Clone the repository and create a Conda environment:
```bash
git clone https://github.com/jumpynitro/POCA.git
conda activate poca
```

### Step 2: Install dependencies

Install the required packages:
```bash
pip install -r requirements.txt
```

---

## 2. Running the Custom Acquisition Process

The acquisition process has three components:
1. **Train the LLM**  
2. **Generate samples with the LLM (MC samples)**  
3. **Train the downstream model (RF or NN) and perform acquisition**  

### Example Experiment

For instance, to train the `mistral` model on the `uci/magic` dataset using a seed of 1, generate 8 MC samples, and perform acquisition using Random Forest (RF) on historical data with metrics like PO-EIG, EIG, Random, and EIG (full feature observation) across 60 seeds, follow these steps:

#### Step 1: Train the LLM

Run the following to train the LLM:
```bash
python unsloth_train_llm.py \
    data="uci/magic" \
    llm_cfg.pool_set_type='Hist' \
    llm_cfg.llm_name='mistral3-unsloth' \
    llm_cfg.llm_dir="LLM_dir" \
    llm_cfg.add_name='-seed1-pdict-true' \
    llm_cfg.llm_seed=1
```

#### Step 2: Generate MC Samples

Run the following to generate MC samples:
```bash
python unsloth_eval_llm.py \
    data="uci/magic" \
    llm_cfg.pool_set_type='Hist' \
    llm_cfg.llm_name='mistral3-unsloth' \
    llm_cfg.batch_size_generation=100 \
    llm_cfg.mc_samples=8 \
    llm_cfg.add_name='-seed1-pdict-true' \
    llm_cfg.llm_dir="LLM_Hist"
```

#### Step 3: Train the RF Model and Run Acquisition Process


1. **Acquistion with GSMs**
    Use the following command to train the RF model and execute the acquisition:
    ```bash
    python main.py --multirun 'rng.seed=range(60)' \
        data="uci/magic" \
        results_main_dir="rf_results_Hist" \
        acquisition.objective=bald-po,bald,random,full-bald \
        model=random_forest \
        trainer=random_forest \
        llm_cfg.llm_name='mistral3-unsloth' \
        llm_cfg.pool_set_type='Hist' \
        llm_cfg.add_name='-seed1-pdict-true' \
        llm_cfg.llm_dir="LLM_Hist"
    ```
    
    For this specific case we can use the following script:

    ```
    python generate_plots/create_plot_general.py --experiment_type specific
    ```

2. **Cost based acquisition**
    To run acquisition with subset of features acquired run:
    ```bash
    python main.py --multirun 'rng.seed=range(60)' \
        data="uci/magic" \
        results_main_dir="rf_results_Hist" \
        acquisition.objective=bald-po-feature-0.2,bald-po-feature-0.6 \
        model=random_forest \
        trainer=random_forest \
        llm_cfg.llm_name='mistral3-unsloth' \
        llm_cfg.pool_set_type='Hist' \
        llm_cfg.add_name='-seed1-pdict-true' \
        llm_cfg.llm_dir="LLM_dir"
    ```

    and for visualizing 

    ```
    python generate_plots/create_plot_cost.py
    ```

---

## 3. Reproducing Paper Results

You can reproduce most of the paper's results by following these steps:

1. **Set the path for results storage**:  
   Specify the path where LLMs, generated samples, and results should be saved:
   ```bash
   export path_mnt=...
   ```

2. **Generate samples from LLMs (GSMs)**:
   - GSM trained on historical data:
     ```bash
     bash jobs/samples_GSM_all_llms.sh
     ```
   - GSM trained on pool data:
     ```bash
     bash jobs/samples_GSM_mistral_pool.sh
     ```

3. **Run models using RF and acquisition metrics**: PO-EIG, EIG, Random, and EIG (all features):
   1. RF training and acquisition on historical data:
     ```bash
     bash jobs/run_rf_eig.sh
     ```

   2. RF training and acquisition on pool data:
     ```bash
     bash jobs/run_rf_eig_pool.sh
     ```

4. **Run additional acquisition metrics**: EPIG, MeanSTD, Marginal Entropy:
   ```bash
   bash jobs/run_rf_other_models.sh
   ```

5. **Run EIG with a subset of features**: Experiment with acquiring only 20% or 60% of features on the `magic` dataset:
   ```bash
   bash jobs/run_rf_cost.sh
   ```
6. **Plots some results
   Results EIG (run 3.1) and EPIG (run 4) ```python generate_plots/create_plot_general.py --experiment_type vary_model_main --output_file model_main.pdf```

   Results Varying LLMs ```python generate_plots/create_plot_general.py --experiment_type vary_llm --output_file llm_main.pdf```

---
## Citation
If our paper or code helped you in your own research, please cite our work as:

```
@inproceedings{
    astorga2024poca,
    title={Partially Observable Cost-Aware Active-Learning with Large Language Models},
    author={Nicol{\'a}s Astorga and Tennison Liu and Nabeel Seedat and Mihaela van der Schaar},
    booktitle={The Thirty-Eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
