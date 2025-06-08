# EFDiscover: Prior-Free Discovery of Inter-Attribute Expression Functional Dependencies in Noisy Data

# Overview
**EFDiscover** is a cutting-edge method for mining Expression Functional Dependencies (EFDs) from noisy data. While the full implementation details remain proprietary, this research provides comprehensive performance benchmarks and detailed experimental results, establishing a new standard for functional dependency discovery. This enables researchers to evaluate and compare the efficiency of their own dependency mining systems across diverse real-world datasets. The benchmark facilitates direct comparisons between EFDiscover's performance, traditional FD mining approaches, and novel techniques, offering a standardized framework for assessing dependency discovery effectiveness in noisy data environments.


The repository includes:
- **Real-world native datasets** used by EFDiscover for testing.
- **Baseline performance logs** for comparison with Uniclean’s results.

# Dataset Information

The following table summarizes the datasets used in this repository, including their error types and dimensions:

| Dataset  | Shape        | Dataset Description                                                                                                     |
|----------|--------------| ------------------------------------------------------------------------------------------------------------------------|
| Foodstamp| 4 × 4        | Statistics on Household Information for Recipients of Food Assistance.                                                  |
| IDF      | 2,000 × 20   | Records of temperature variations in lubricating oil across various components of a wind turbine lubrication system.    |
| Glass    | 213 × 11     | Records the oxide content in six types of glass, including elements (i.e., Na, Fe, K, etc).                             |
| Abalone  | 4,177 × 8    | Measure the age of abalone through indicators such as length and weight.                                                |
| Rice     | 3,808 × 6    | Data recording of morphological characteristics of rice.                                                                |



# Running Uniclean’s Cleaning Performance Test

To evaluate Uniclean’s cleaning performance, run the `run.sh` script. This script automates the cleaning process across all datasets and saves performance logs in the `Uniclean_logs/` directory.

## Usage
```bash
# Give execution permissions
chmod +x run.sh

# Run the script
./run.sh
```

The `run.sh` script iterates over each dataset in the `datasets/original_datasets/` directory, processes it with Uniclean, and logs the results. Each dataset has its specific configuration, including `mse_attributes` (attributes for Mean Squared Error calculation) and `elapsed_time` parameters. The results of each dataset’s cleaning process are saved in the corresponding subdirectory within `Uniclean_logs/`.

# Cleaners  Library Overview

## uniclean_cleaners/SampleScrubber
**Sample Cleaning Tools**
- **ModuleTest**: Unit tests for modules.
- **util**
    - `distance.py`: Computes distances between values.
    - `getNum.py`: Evaluates cleaning accuracy.
- `uniop_model.py`: Rule mining model.
- `param_builder.py`: Constructs rule parameters.
- `param_selector.py`: Selects optimal parameters.
- **cleaners**
    - `single.py`: Single-attribute operators.
    - `multiple.py`: Multi-attribute relational operators.
    - `soft.py`: Experimental or soft operators.
    - `clean_penalty.py`: Calculates cleaning costs (edit distance, semantic penalties, Jaccard penalties).


## Conguration script in ./uniclean_cleaners
- `main.py`: Command-line entry point for one-click data cleaning.
- `logsetting.py`: Logging configuration for the one-click pipeline.
- `Clean.py`: Core script for terminal-based cleaning logic.
- `requirements.txt`: Dependency list for the one-click cleaning system.
- `Plantuml.svg`: Flowchart visualizing the cleaning pipeline.

# Repository Structure
- `datasets_and_rules/`:real word datasets、inject error datasets and their cleaning rules:
  - `artificial_error_datasets/`:Contains datasets with artificially injected errors in eight different proportions (ranging from 0.25% to 2%) for controlled experiments and benchmarking. This folder also includes the *BART script* used for injecting these errors into the datasets.
  - `original_datasets/`: Contains real-world datasets in their native (uncleaned) form.
- `Uniclean_cleaned_data/`: Datasets that have been cleaned by Uniclean.
  - `artificial_error_cleaned_data/`:Uniclean-cleaned versions of the artificially injected error datasets.
  - `original_error_cleaned_data/`:Uniclean-cleaned  versions of the real-world datasets containing native errors.
- `Uniclean_cleaner_workflow_logs/`: Logs generated during the Uniclean cleaning process and Cleaner attributes dependencies for each dataset.
  - `artificial_error_cleaner_workflow_logs/`: Step-by-step workflow logs for datasets that had artificial errors (in different proportions).
  - `original_error_cleaner_workflow_logs/`:Step-by-step workflow logs for real-world datasets with native errors.
- `Uniclean_results/`: Contains the final outputs and performance metrics from Uniclean’s data cleaning for each dataset.
  - `artificial_error_results/`:Final outputs and metrics (e.g., accuracy, F1 score) from Uniclean’s cleaning for datasets that had artificially injected errors in different proportions.
  - `original_error_results/`:Final outputs and metrics from Uniclean’s cleaning for real-world datasets containing native errors.
- `baseline_cleaning_systems_logs/`: Logs documenting the performance of baseline systems on the same datasets, enabling a direct comparison with Uniclean’s results.
  - `artificial_error_datasets/`:Stores log files showing how baseline systems perform on datasets with artificial errors.
    - **File Naming Format**: `[dataset_name]_[cleaning_system_name]_nwcpk_[error_proportion].log`
    - Example: `1_hospitals_raha_baran_nwcpk_1.log`
  - `original_datasets/`:Stores log files showing how baseline systems perform on real-world datasets with native errors.
    - **File Naming Format**: `[dataset_name]_ori_[cleaning_system_name]_[the actual size of the dataset (if it is not in its original size)].log`
    - Example: `1_hospital_ori_baran.log`
- `baseline_cleaning_systems_results/`: Final results and performance metrics of baseline systems on the same datasets.
  - `artificial_error_datasets/`:Contains overall performance metrics (e.g., accuracy, recall, F1 score) of baseline systems on artificially injected error datasets.
    - **Folder Naming Format**: `[dataset_name]_nwcpk_[error_proportion]`
    - Example: `1_hospitals_nwcpk_1`
  - `original_datasets/`:Contains overall performance metrics of baseline systems on real-world datasets with native errors.
    - **Folder Naming Format**: `[dataset_name]_[the actual size of the dataset (if it is not in its original size)]_ori`
    - Example: `1_hospital_ori`
- `baseline_cleaned_data/`:Datasets that have been cleaned by baseline systems.
  - `artificial_error_datasets/`:Baseline-cleaned versions of artificially injected error datasets.
    - **File Naming Format**: `[dataset_name]_[error_proportion]_cleaned_by_[cleaning_system_name].csv`
    - Example: `1_hospitals_1_cleaned_by_baran.csv`
  - `original_datasets/`:Baseline-cleaned versions of real-world datasets with native errors.
    - **File Naming Format**: `[dataset_name][the actual size of the dataset (if it is not in its original size)]_cleaned_by_[cleaning_system_name].csv`
    - Example: `1_hospital_cleaned_by_baran.csv`
- `evaluate_result.py`: A script that computes performance metrics for data cleaning, such as accuracy, recall, F1 score, and error reduction rate, allowing comprehensive evaluation of data cleaning effectiveness.
- `get_holoclean_table.py` A script that transforms datasets into the Holoclean-compatible input CSV format. It transposes data and ensures compliance with Holoclean's required schema for further data cleaning tasks.
- `get_error_num.py` A script that compares dirty data with clean data to compute the number of erroneous cells and entries. It provides a detailed analysis of the extent of errors, facilitating error quantification and benchmarking.
