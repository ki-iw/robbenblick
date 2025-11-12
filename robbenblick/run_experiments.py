import subprocess
import yaml
from itertools import product
import copy
import os
import pandas as pd
import argparse
from pathlib import Path

from robbenblick import logger, CONFIG_PATH, PROJECT_ROOT
from robbenblick.utils import load_config

DEFAULT_CONFIG_PATH = CONFIG_PATH / "base_iter_config.yaml"
TRAIN_SCRIPT = "robbenblick/yolo.py"


# --- Helper Functions ---

def find_iterables(config_dict):
    """
    Recursively searches the config dictionary for lists,
    which are treated as iterable parameters (e.g., [10, 20]).
    Returns a "flat" dictionary with dot-separated keys.

    Example:
    Input: {'a': 1, 'b': {'c': [10, 20]}}
    Output: {'b.c': [10, 20]}
    """
    iterables = {}

    def recurse(d, path=""):
        for key, value in d.items():
            new_path = f"{path}.{key}" if path else key

            if isinstance(value, list):
                # We assume every list is an iterable parameter
                iterables[new_path] = value
                logger.info(f"Found iterable parameter: {new_path} = {value}")
            elif isinstance(value, dict):
                # Recurse into sub-dictionaries
                recurse(value, new_path)

    recurse(config_dict)
    return iterables


def generate_variants(iterables):
    """
    Takes the dictionary of iterables and creates a
    list of "variants," where each variant is a specific
    combination of parameters.

    Input: {'b.c': [10, 20], 'd': [True, False]}
    Output:
    [
        {'b.c': 10, 'd': True},
        {'b.c': 10, 'd': False},
        {'b.c': 20, 'd': True},
        {'b.c': 20, 'd': False}
    ]
    """
    if not iterables:
        return [{}]  # One run with the base configuration

    keys, values = zip(*iterables.items())

    # 'product' creates the Cartesian product of all parameter combinations
    value_combinations = list(product(*values))

    variants = []
    for combo in value_combinations:
        variant = dict(zip(keys, combo))
        variants.append(variant)

    return variants


def set_nested_key(d, key_path, value):
    """
    Sets a value in a nested dictionary based on
    a dot-separated key path.

    Example: set_nested_key({}, 'b.c', 10) -> {'b': {'c': 10}}
    """
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def run_experiment(base_config, variant, base_run_id):
    """
    Executes a single experiment run (data creation + training)
    for a given parameter variant.
    """
    # 1. Create a deep copy of the base configuration
    variant_config = copy.deepcopy(base_config)

    # 2. Create a unique run name for this variant
    variant_suffix = []
    for key, value in variant.items():
        # Replace '.' with '_' for cleaner names (e.g., 'yolo_hyp_batch')
        clean_key = key.replace('.', '_')
        if clean_key.startswith("yolo_hyperparams_"):
            clean_key = clean_key.replace("yolo_hyperparams_", "")
        variant_suffix.append(f"{clean_key}_{value}")

    if not variant_suffix:
        # This is the base run (no variants)
        run_id = base_run_id
    else:
        run_id = f"{base_run_id}_{'_'.join(variant_suffix)}"

    logger.info("=" * 50)
    logger.info(f"--- STARTING EXPERIMENT: {run_id} ---")

    # 3. Update the configuration with the variant's values
    variant_config['run_id'] = run_id
    for key, value in variant.items():
        set_nested_key(variant_config, key, value)
        logger.info(f"  Override: {key} = {value}")

    # 4. Define output path and save the temporary config file
    # We save the config *inside* the run's future output directory
    run_output_dir = PROJECT_ROOT / "runs" / "detect" / run_id

    # Create this directory now so we can write the config file
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Define the path for the temp config *inside* the run directory
    temp_config_path = run_output_dir / f"config_for_run_{run_id}.yaml"

    with open(temp_config_path, 'w') as f:
        yaml.dump(variant_config, f, sort_keys=False)

    logger.info(f"Temporary config saved to: {temp_config_path}")

    try:
        # Run yolo.py (Training) ---
        logger.info(f"Running yolo.py (train) for {run_id}...")
        # Set up the environment for the subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)

        project_save_dir = run_output_dir.parent

        # We pass '--run_id' explicitly to ensure
        # yolo.py uses the name we generated here.
        subprocess.run(
            [
                "python",
                TRAIN_SCRIPT,
                "--config", str(temp_config_path),
                "--mode", "train",
                "--run_id", run_id,
                "--project-dir", str(project_save_dir)
            ],
            check=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env
        )
        logger.info(f"--- EXPERIMENT {run_id} COMPLETED SUCCESSFULLY ---")

        # Automatic evaluation ---
        logger.info(f"--- EVALUATING RESULTS for {run_id} ---")
        try:
            results_path = PROJECT_ROOT / "runs" / "detect" / run_id / "results.csv"

            if not results_path.exists():
                logger.warning(f"Evaluation skipped: results.csv not found at {results_path}")
                return None
            else:
                df = pd.read_csv(results_path)
                # Clean column names (they often have whitespace)
                df.columns = df.columns.str.strip()

                # Find the best epoch based on mAP50(B)
                best_epoch_data = df.sort_values(by="metrics/mAP50(B)", ascending=False).iloc[0]

                # Extract key metrics
                result_dict = {
                    "run_name": run_id,
                    "epoch": int(best_epoch_data["epoch"]),
                    "mAP50": best_epoch_data["metrics/mAP50(B)"],
                    "mAP50-95": best_epoch_data["metrics/mAP50-95(B)"],
                    "precision": best_epoch_data["metrics/precision(B)"],
                    "recall": best_epoch_data["metrics/recall(B)"]
                }
                logger.info("--- BEST EPOCH PERFORMANCE ---")
                logger.info(f"  Run Name: {run_id}")
                logger.info(f"  BEST EPOCH (Epoch {result_dict['epoch']}):")
                logger.info(f"    mAP50:      {result_dict['mAP50']:.4f}")
                logger.info(f"    mAP50-95:   {result_dict['mAP50-95']:.4f}")

                return result_dict
        except Exception as eval_e:
            logger.error(f"Failed to evaluate results for {run_id}: {eval_e}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during experiment {run_id}:")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"--- EXPERIMENT {run_id} FAILED ---")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Hyperparameter Experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the base YAML config file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=-1,
        help="Display Top N runs. Use -1 to display all."
    )
    args = parser.parse_args()

    logger.info("Starting hyperparameter experiment run...")

    base_config =  load_config(args.config)
    if base_config is None:
        exit(1)

    base_run_id = base_config.get('run_id', 'experiment')

    # Find iterable parameters (also nested)
    iterables = find_iterables(base_config)

    if not iterables:
        logger.warning("No iterable parameters (lists) found in config.")
        logger.warning("Running a single pass with the base configuration.")

    # Generate all variant combinations
    variants = generate_variants(iterables)

    logger.info(f"Found a total of {len(variants)} experiment variants.")

    all_results = []

    # Run each variant as a separate experiment
    for i, variant in enumerate(variants):
        logger.info(f"Starting variant {i + 1} of {len(variants)}")
        best_epoch_result = run_experiment(base_config, variant, base_run_id)
        if best_epoch_result:
            all_results.append(best_epoch_result)

    logger.info("=" * 50)
    logger.info("All experiments completed. 🚀")

    if not all_results:
        logger.warning("No successful runs to evaluate.")
    else:
        logger.info("--- 🏆 FINAL EXPERIMENT RANKING 🏆 ---")

        # Create dataframe from all results
        final_df = pd.DataFrame(all_results)

        # Sort by mAP50(B)
        final_df_sorted = final_df.sort_values(by="mAP50", ascending=False)

        # Show Top N runs
        n_top_runs = min(args.top_n, len(final_df_sorted)) if args.top_n > 0 else len(final_df_sorted)
        logger.info(f"Top {n_top_runs} runs sorted by mAP50(B):")
        columns_to_show = ["run_name", "mAP50", "mAP50-95", "precision", "recall", "epoch"]
        # use print instead of logger to format the table properly
        print(
            final_df_sorted[columns_to_show].head(n_top_runs).to_string(index=False)
        )

if __name__ == "__main__":
    main()