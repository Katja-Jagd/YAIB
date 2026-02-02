import json
import os

#data_path = "/isdata/winthergrp/gsn245/scratch/YAIB/pretrained_checkpoints/grud_supervised_mimic_mortality"
model_name = "DeepSetAttention"
dataset_name = "hirid"
#dataset_name = "mimic"
#data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_grud_supervised_{dataset_name}_mortality/{dataset_name}/Mortality24/{model_name}"
#data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_grud_supervised_{dataset_name}_aki_2/{dataset_name}/AKI/{model_name}"
#data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_grud_supervised_{dataset_name}_mortality_test_2/{dataset_name}/Mortality24/{model_name}"
#data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_bat_supervised_{dataset_name}_aki_test/{dataset_name}/AKI/{model_name}"
#data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_itransformer_supervised_hirid_mortality/hirid/Mortality24/iTransformer"
#data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_ipnets_supervised_hirid_mortality/hirid/Mortality24/IPNets"
data_path = f"/isdata/winthergrp/gsn245/scratch/yaib_logs_deep_set_attention_supervised_hirid_mortality/hirid/Mortality24/DeepSetAttention"



train_sizes=[100, 500, 1000, 2000, 3000, 5000, 7000, 9000, 9506]
seeds=[42, 84, 126, 168, 210]
file_name = "aggregated_test_metrics.json"

def check_duplicate_folders(training_params, dir_list):
    """
    Check if there are any duplicate training parameters (train_size, seed) combinations.
    """
    duplicates = []
    seen = {}

    for params, dir_name in zip(training_params, dir_list):
        if params in seen:
            duplicates.append({
                'params': params,
                'folders': [seen[params], dir_name]
            })
            print(f"WARNING: Duplicate training parameters found!")
            print(f"  Parameters: train_size={params[0]}, seed={params[1]}")
            print(f"  Folders: {seen[params]} and {dir_name}")
        else:
            seen[params] = dir_name

    return duplicates


def map_dir_to_params(data_path, train_sizes=None, seeds=None, use_folds=False):
    """
    Walk through directories and read train_config.gin to extract actual training parameters.
    Creates a mapping from (train_size, seed) to directory name.
    Also checks for and reports any duplicate parameter combinations.

    Args:
        data_path: Path to the directory containing experiment results
        train_sizes: Optional list of expected training sizes (not used currently)
        seeds: Optional list of expected seeds (not used currently)
        use_folds: If True, includes all folds (0-4) in average calculations. If False (default),
                  only uses fold_0 for each directory.
    """
    # Get all directories, sorted by creation time
    dir_list = os.listdir(data_path)
    dir_list.sort(key=lambda x: os.path.getctime(os.path.join(data_path, x)))

    # Remove excluded directories
    excluded = ["old", "sweep"]
    for excluded_dir in excluded:
        if excluded_dir in dir_list:
            dir_list.remove(excluded_dir)

    # Extract training parameters from each directory
    training_params = []
    valid_dirs = []

    for dir_name in dir_list:
        config_file = os.path.join(data_path, dir_name, "repetition_0", "fold_0", "train_config.gin")

        try:
            with open(config_file, "r") as f:
                lines = f.readlines()
                seed = None
                train_size = None

                for line in lines:
                    if "execute_repeated_cv.subset_train_seed" in line:
                        seed = int(line.split('=')[1].strip())
                    if "execute_repeated_cv.subset_train_size" in line:
                        train_size = int(line.split('=')[1].strip())

                if seed is not None and train_size is not None:
                    training_params.append((train_size, seed))
                    valid_dirs.append(dir_name)
                    print(f"Directory: {dir_name}, Train Size: {train_size}, Seed: {seed}")
                else:
                    print(f"WARNING: Could not extract parameters from {dir_name}")
        except FileNotFoundError:
            print(f"WARNING: Config file not found for {dir_name}, skipping")
        except Exception as e:
            print(f"ERROR: Failed to process {dir_name}: {e}")

    # Check for duplicates
    duplicates = check_duplicate_folders(training_params, valid_dirs)
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate parameter combination(s)!")
        print("Using only the most recent directory for each (train_size, seed) combination")
    else:
        print("\nNo duplicate training parameters found.")

    # Create parameter-based mapping
    # Only keep the most recent directory for each parameter combination
    # Since dir_list is sorted by creation time, dict() will keep the last (most recent) one
    dir_map = dict(zip(training_params, valid_dirs))

    # Create size-only mapping (maps train_size -> list of directories)
    dir_map_by_size = {}
    for (train_size, seed), dir_name in dir_map.items():
        if train_size not in dir_map_by_size:
            dir_map_by_size[train_size] = []
        dir_map_by_size[train_size].append(dir_name)

    print(f"\nParameter mapping created: {len(dir_map)} directories")
    print(f"Size-based grouping: {dict((k, len(v)) for k, v in dir_map_by_size.items())}")

    return dir_map, dir_map_by_size


def calc_avg_performance(data_path, dir_map_by_size, train_size, file_name, use_folds=False):
    """
    Calculate average performance metrics for all experiments with a given train size.

    Args:
        data_path: Path to the directory containing experiment results
        dir_map_by_size: Dictionary mapping train_size to list of directory names
        train_size: The training size to calculate metrics for
        file_name: Name of the metrics file to read
        use_folds: If True, averages across folds 0-4 for each directory before computing
                  the overall average. If False, only uses fold_0.
    """
    # Get all directories that match the train_size
    matching_dirs = dir_map_by_size.get(train_size, [])

    if not matching_dirs:
        print(f"WARNING: No directories found for train_size={train_size}")
        return None, None, None, None, None, None

    auprc_list = []
    auroc_list = []
    loss_list = []
    repetitions = 0

    for dir_name in matching_dirs:
        print(f"Processing directory: {dir_name}")
        file_name_full = f"{data_path}/{dir_name}/{file_name}"
        try:
            with open(file_name_full, "r") as f:
                metrics = json.load(f)

                if use_folds:
                    # Include all folds 0-4 from this directory in the overall average
                    fold_count = 0

                    for fold_idx in range(5):
                        fold_key = f"fold_{fold_idx}"
                        try:
                            auroc = metrics["repetition_" + str(repetitions)][fold_key]["AUC"] * 100
                            auprc = metrics["repetition_" + str(repetitions)][fold_key]["PR"] * 100
                            loss = metrics["repetition_" + str(repetitions)][fold_key]["loss"] * 100
                            auroc_list.append(auroc)
                            auprc_list.append(auprc)
                            loss_list.append(loss)
                            fold_count += 1
                        except KeyError:
                            print(f"  WARNING: {fold_key} not found in {dir_name}, skipping this fold")

                    if fold_count > 0:
                        print(f"  Repetition {repetitions}: Included {fold_count} folds")
                    else:
                        print(f"  WARNING: No valid folds found for {dir_name}")
                else:
                    # Use only fold_0
                    fold_key = f"fold_0"
                    auroc = metrics["repetition_" + str(repetitions)][fold_key]["AUC"] * 100
                    auprc = metrics["repetition_" + str(repetitions)][fold_key]["PR"] * 100
                    loss = metrics["repetition_" + str(repetitions)][fold_key]["loss"] * 100
                    auroc_list.append(auroc)
                    auprc_list.append(auprc)
                    loss_list.append(loss)
                    print(f"  Repetition {repetitions}: AUROC={auroc:.2f}, AUPRC={auprc:.2f}")

        except FileNotFoundError:
            print(f"  WARNING: Metrics file not found for {dir_name}, skipping")
        except Exception as e:
            print(f"  ERROR: Failed to process metrics for {dir_name}: {e}")

    if not auroc_list:
        print(f"WARNING: No valid metrics found for train_size={train_size}")
        return None, None, None, None, None, None

    avg_auroc = sum(auroc_list) / len(auroc_list)
    avg_auprc = sum(auprc_list) / len(auprc_list)
    avg_loss = sum(loss_list) / len(loss_list)
    std_auroc = (sum((x - avg_auroc) ** 2 for x in auroc_list) / len(auroc_list)) ** 0.5
    std_auprc = (sum((x - avg_auprc) ** 2 for x in auprc_list) / len(auprc_list)) ** 0.5
    std_loss = (sum((x - avg_loss) ** 2 for x in loss_list) / len(loss_list)) ** 0.5
    return avg_auprc, avg_auroc, avg_loss, std_auprc, std_auroc, std_loss


def main(use_folds=False):
    """
    Main function to calculate average performance across experiments.

    Args:
        use_folds: If True, averages across folds 0-4 for each parameter combination
                  before computing the overall average. If False (default), only uses fold_0.
    """
    dir_map, dir_map_by_size = map_dir_to_params(data_path, train_sizes, seeds, use_folds)

    # Get actual train sizes from discovered directories
    discovered_train_sizes = sorted(dir_map_by_size.keys())
    print(f"\nDiscovered train sizes: {discovered_train_sizes}")

    sample_sizes = {}
    for train_size in discovered_train_sizes:
        avg_auprc, avg_auroc, avg_loss, std_auprc, std_auroc, std_loss = calc_avg_performance(
            data_path, dir_map_by_size, train_size, file_name, use_folds
        )

        if avg_auroc is not None:
            sample_sizes[str(train_size)] = {
                "test/AUC": f"{avg_auroc:.2f} ± {std_auroc:.2f}",
                "test/PR": f"{avg_auprc:.2f} ± {std_auprc:.2f}",
                "test/loss": f"{avg_loss:.2f} ± {std_loss:.2f}"
            }
            print(f"\nTrain size {train_size}: AUROC={avg_auroc:.2f} ± {std_auroc:.2f}, "
                  f"AUPRC={avg_auprc:.2f} ± {std_auprc:.2f}, Loss={avg_loss:.2f} ± {std_loss:.2f}")
        else:
            print(f"\nTrain size {train_size}: No valid metrics found")

    model = model_name
    dataset = dataset_name
    type = "subset"
    summary = {
        "model": model,
        "dataset": dataset,
        "type": type,
        "sample_sizes": sample_sizes
    }
    output_file = f"{data_path}/average_performance_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
        print(f"\nSaved average performance summary to {output_file}")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate average performance metrics across experiments")
    parser.add_argument(
        "--folds",
        action="store_true",
        help="Average across folds 0-4 for each parameter combination before computing the overall average. By default, only uses fold_0."
    )
    args = parser.parse_args()

    main(use_folds=args.folds)