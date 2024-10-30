import hydra
import logging
import numpy as np
import os
import torch
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from pathlib import Path
from src.utils import Dictionary, get_repo_status, format_time
import time
import random

# Absolute path for the current working directory
ABS_PATH = os.path.abspath('.')

# Utility function to create a directory if it doesn't exist
def create_path_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    # Random number generator and device setup (GPU if available)
    rng = call(cfg.rng)
    device = "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"

    # Process acquisition objectives with 'full-' prefix
    if 'full-' in cfg.acquisition.objective:
        cfg.partially_observed = False
        cfg.acquisition.objective = cfg.acquisition.objective.replace('full-', '')

    is_first_al_step = True
    # Logging setup
    time.sleep(0.2)
    logging.info(f"Seed: {cfg.rng.seed}")
    logging.info(f"Device: {device}")

    # PO (Partially Observed) configuration
    PO_DUMB_IMPUT_NOT_ACQ = cfg.po_dumb_imput_not_acq
    PO_DUMB_IMPUT_ACQ = cfg.po_dumb_imput_acq
    PO_DUMB_IMPUT = PO_DUMB_IMPUT_NOT_ACQ or PO_DUMB_IMPUT_ACQ
    if PO_DUMB_IMPUT:
        cfg.partially_observed = True

    # Load the data and transfer to device
    logging.info("Loading data")
    data = instantiate(cfg.data, rng=rng, seed=cfg.rng.seed, partially_observed=cfg.partially_observed, llm_cfg=cfg.llm_cfg)
    data.torch()
    data.to(device)

    # Set variables related to dataset and masking
    features_idx_unobs = None
    use_mask = data.main_dataset.use_mask
    this_dataset_name = data.main_dataset.this_dataset_name
    mask_input_used = None

    # Add additional name info based on dataset and PO configuration
    if 'syndata' in this_dataset_name:
        add_exp_name = f"_{cfg.llm_cfg.syn_class}"
    else:
        add_exp_name = ''

    if PO_DUMB_IMPUT_ACQ:
        add_exp_name += '_DB_IMP_AQ'
    if PO_DUMB_IMPUT_NOT_ACQ:
        add_exp_name += '_DB_IMP_NOTAQ'

    # MC (Monte Carlo) samples handling
    add_name_mc_samples = ''
    if cfg.llm_cfg.use_name_mc_samples:
        add_name_mc_samples = f'_MC{cfg.llm_cfg.max_mc_samples}'

    # Final experimental name based on observation type
    if cfg.partially_observed:
        experimental_name_used = f"{cfg.acquisition.objective}{add_exp_name}{cfg.llm_cfg.add_name_results}{add_name_mc_samples}_{cfg.feature_selection.imputation_type}"
    else:
        experimental_name_used = f"{cfg.acquisition.objective}{add_exp_name}{cfg.llm_cfg.add_name_results}{add_name_mc_samples}_{cfg.feature_selection.imputation_type}_FO"
        use_mask = False

    # Additional processing for acquisition objectives
    if cfg.acquisition.objective == 'bald' and use_mask and not PO_DUMB_IMPUT:
        cfg.acquisition.objective = 'bald-po-bald-marginalized'
    if cfg.acquisition.objective == 'epig' and use_mask and not PO_DUMB_IMPUT:
        cfg.acquisition.objective = 'epig-po-epig-marginalized'

    # Set up result directories
    logging.info(f"Making results directories at {cfg.directories.results_run}")
    if cfg.results_main_dir is None:
        results_dir = Path(f"{ABS_PATH}/results/{data.main_dataset.get_dir_llm_vanilla()}/{experimental_name_used}/seed{cfg.rng.seed}")
    else:
        results_dir = Path(f"{cfg.results_main_dir}/{data.main_dataset.get_dir_llm_vanilla()}/{experimental_name_used}/seed{cfg.rng.seed}")
    
    create_path_if_not_exists(results_dir)

    # Modification Start: Check if the run has already been completed
    completion_file = results_dir / "run_completed.csv"
    if completion_file.exists():
        print("Run already completed for this configuration. Skipping execution.")
        return
    
    logging.info(f"Number of classes: {data.n_classes}")

    # Load LLM-generated data if PO is enabled and acquisition objective contains '-po'
    if "-po" in cfg.acquisition.objective and not PO_DUMB_IMPUT:
        logging.info("Loading LLM-generated samples")
        # Generate a random sleep time between 0 and 5 seconds
        sleep_time = random.uniform(0, 10)
        # Sleep for the random time
        time.sleep(sleep_time)
        
        gen_data = data.load_llm_samples(is_numpy=True, get_original=False, filter_unobs=True, apply_mode=True,
                                         max_mc_samples=cfg.llm_cfg.max_mc_samples,
                                         max_mc_samples_random=cfg.llm_cfg.max_mc_samples_random)

        logging.info(f"Loading {gen_data.shape[1]} MC samples from LLM")
        gen_data = torch.Tensor(gen_data).to(device)
        gen_data = data.normalize_data(gen_data, partially_observed=True)

        if "bald-po-feature" in cfg.acquisition.objective:
            features_idx_unobs = data.get_features_indexes_matrix()

        gen_data[torch.isinf(gen_data)] = 0
        data.update_gen_data(gen_data)

        if use_mask:
            mask_table, mask_table_extended = data.main_dataset.get_original_mask_pool_data()
            mask_input_used = torch.Tensor(mask_table_extended.to_numpy().astype(float)).to(device)
            data.update_mask_data(torch.Tensor(mask_table.to_numpy().astype(float)).to(device), mask_input_used)

    if PO_DUMB_IMPUT and use_mask:
        logging.info("Applying dumb imputation with mask")
        mask_table, mask_table_extended = data.main_dataset.get_original_mask_pool_data()
        mask_input_used = torch.Tensor(mask_table_extended.to_numpy().astype(float)).to(device)
        data.update_mask_data(torch.Tensor(mask_table.to_numpy().astype(float)).to(device), mask_input_used)
        gen_data = mask_input_used * 0
        data.update_gen_data(gen_data)

    # Load cost data
    cost_data = data.load_cost_data(cfg.feature_selection.include_y_cost,
                                    cfg.feature_selection.different_x_cost,
                                    cfg.feature_selection.stochastic_cost)
    cost_summed = cost_data.sum(1)
    if cfg.feature_selection.is_cost_based:
        data.update_cost_data(torch.Tensor(cost_data).to(device))

    # Active learning setup
    logging.info("Starting active learning")
    IS_COST_RESTRICTION = cfg.feature_selection.cost_restriction
    MAX_BUDGET = cfg.llm_cfg.max_budget
    acum_cost = 0

    test_log = Dictionary()
    cost_log = Dictionary()
    if cfg.partially_observed and not use_mask:
        test_log_po = Dictionary()

    PUT_ADD_SCORES = False
    if 'syndata' in this_dataset_name:
        mean_score_metrics = Dictionary()
        if 'bald-po' == cfg.acquisition.objective or 'bald-po-marginal' == cfg.acquisition.objective:
            diff_mean_score_metrics = Dictionary()
            PUT_ADD_SCORES = True

    # Active learning loop
    while True:
        if IS_COST_RESTRICTION and acum_cost > MAX_BUDGET:
            break

        n_labels_str = f"{data.n_train_labels:04}_labels"
        is_last_al_step = data.n_train_labels >= cfg.acquisition.n_train_labels_end

        logging.info(f"Number of labels: {data.n_train_labels}")

        # Trainer setup
        logging.info("Setting up trainer")
        if cfg.model_type == "nn":
            model = instantiate(cfg.model, input_shape=torch.Size((data.input_shape_total,)), output_size=data.n_classes).to(device)
            trainer = instantiate(cfg.trainer, model=model)

            if cfg.partially_observed and not use_mask:
                model_po = instantiate(cfg.model, input_shape=torch.Size((data.input_shape_po,)), output_size=data.n_classes).to(device)
                trainer_po = instantiate(cfg.trainer, model=model_po)

        elif cfg.model_type == "rf":
            model = instantiate(cfg.model)
            trainer = instantiate(cfg.trainer, model=model)

            if cfg.partially_observed and not use_mask:
                model_po = instantiate(cfg.model)
                trainer_po = instantiate(cfg.trainer, model=model_po)

        else:
            raise ValueError("Unsupported model type")

        # Model training
        if data.n_train_labels > 0:
            logging.info("Training the model")
            train_log = trainer.train(
                train_loader=data.get_loader("train"),
                val_loader=data.get_loader("val"),
            )

            if cfg.partially_observed and not use_mask:
                train_log_po = trainer_po.train(
                    train_loader=data.get_loader("train", partially_observed=True),
                    val_loader=data.get_loader("val", partially_observed=True),
                )

            if train_log:
                logging.info("Saving training logs")
                formatting = {
                    "step": "{:05}".format,
                    "time": format_time,
                    "train_acc": "{:.4f}".format,
                    "train_loss": "{:.4f}".format,
                    "val_acc": "{:.4f}".format,
                    "val_loss": "{:.4f}".format,
                }
                train_log.save_to_csv(results_dir / "training" / f"{n_labels_str}.csv", formatting)
                if cfg.partially_observed and not use_mask:
                    train_log_po.save_to_csv(results_dir / "training_po" / f"{n_labels_str}.csv", formatting)

        # Model testing
        logging.info("Testing the model")
        with torch.inference_mode():
            test_acc, test_loss = trainer.test(data.get_loader("test"))
            if cfg.partially_observed and not use_mask:
                test_acc_po, test_loss_po = trainer_po.test(data.get_loader("test", partially_observed=True))

        test_log_update = {
            "n_labels": data.n_train_labels,
            "test_acc": test_acc.item(),
            "test_loss": test_loss.item(),
        }
        test_log.append(test_log_update)
        formatting = {"n_labels": "{:04}".format, "test_acc": "{:.4f}".format, "test_loss": "{:.4f}".format}
        test_log.save_to_csv(results_dir / "testing.csv", formatting)
        features_selected = None

        if cfg.partially_observed and not use_mask:
            test_log_po_update = {
                "n_labels": data.n_train_labels,
                "test_acc": test_acc_po.item(),
                "test_loss": test_loss_po.item(),
            }
            test_log_po.append(test_log_po_update)
            test_log_po.save_to_csv(results_dir / "testing_po.csv", formatting)

        # Check if cost restrictions apply or if it's the last active learning step
        if not IS_COST_RESTRICTION and is_last_al_step:
            logging.info("Stopping active learning")
            break

        # Acquisition step based on acquisition objective
        if cfg.acquisition.objective == "random":
            logging.info("Randomly sampling data indices")
            acquired_pool_inds = rng.choice(len(data.inds["pool"]), size=cfg.acquisition.batch_size, replace=False)

        else:
            logging.info("Estimating uncertainty")
            trainer_scorer = trainer if not cfg.partially_observed or "-po" in cfg.acquisition.objective or PO_DUMB_IMPUT else trainer_po

            target_loader = data.get_loader(
                "target", partially_observed=cfg.partially_observed,
                obtain_gen_data=True if "-po" in cfg.acquisition.objective or PO_DUMB_IMPUT else False,
                obtain_cost_data=cfg.feature_selection.is_cost_based, use_mask=use_mask)

            batch_aux = next(iter(target_loader))

            with torch.inference_mode():
                scores = trainer_scorer.estimate_uncertainty(
                    pool_loader=data.get_loader("pool", partially_observed=cfg.partially_observed,
                                                obtain_gen_data=True if "-po" in cfg.acquisition.objective or PO_DUMB_IMPUT else False,
                                                obtain_cost_data=cfg.feature_selection.is_cost_based,
                                                use_mask=use_mask),
                    batch_target_inputs=batch_aux,
                    mode=cfg.acquisition.objective, rng=rng,
                    epig_probs_target=cfg.acquisition.epig_probs_target,
                    epig_probs_adjustment=cfg.acquisition.epig_probs_adjustment,
                    epig_using_matmul=cfg.acquisition.epig_using_matmul,
                    features_idx_unobs=features_idx_unobs,
                    num_subset_pool=cfg.feature_selection.num_subset_pool,
                    dumb_imput=PO_DUMB_IMPUT
                )

            if 'features_selected' in scores:
                features_selected = scores.pop('features_selected')

            if cfg.feature_selection.is_cost_based:
                cost_selected = scores.pop('cost_selected')
                
            scores_obj = scores
            scores = scores.numpy()
            if 'syndata' in this_dataset_name:
                mean_score_metrics.append({"metric_mean": scores[cfg.acquisition.objective].mean()})
                mean_score_metrics.save_to_csv(results_dir / "mean_score.csv", {"metric_mean": "{:.4f}".format})
                if PUT_ADD_SCORES:
                    diff_mean_score_metrics.append({"diff_metrics_mean": (scores['second_MI'] - scores['first_MI']).mean()})
                    diff_mean_score_metrics.save_to_csv(results_dir / "diff_metrics_mean.csv", {"diff_metrics_mean": "{:.4f}".format})

            scores = scores[cfg.acquisition.objective]
            acquired_pool_inds = np.argmax(scores)

        if PUT_ADD_SCORES and (is_first_al_step or not (data.n_train_labels % 5)):
            scores_obj.save_to_csv(results_dir / f"scores-{data.n_train_labels}.csv")

        logging.info(f"Acquiring data with {cfg.acquisition.objective}")
        data.move_from_pool_to_train(acquired_pool_inds, features_selected=features_selected,
                                     imputation_type=cfg.feature_selection.imputation_type,
                                     mask_inputs=mask_input_used, po_dumb_imput_not_acq=PO_DUMB_IMPUT_NOT_ACQ)

        is_first_al_step = False

        if not isinstance(acquired_pool_inds, (list, np.ndarray)):
            pool_inds_to_move = [acquired_pool_inds]
        else:
            pool_inds_to_move = acquired_pool_inds.copy()

        for this_idx in pool_inds_to_move:
            if cfg.feature_selection.is_cost_based:
                acum_cost += float(cost_selected[this_idx])
            else:
                acum_cost += float(cost_summed[this_idx])

        cost_log.append({"cost": acum_cost})
        cost_log.save_to_csv(results_dir / "cost_log.csv", {"cost": "{:.4f}".format})
        time.sleep(0.1)

    completion_file.touch()

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace in case of an error
    main()