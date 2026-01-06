"""IO utilities for saving and loading results."""

import json
import csv
import pickle
from typing import Dict, List, Any
import os


def save_json(data: Dict, filepath: str):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Data saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def save_csv(data: List[Dict[str, Any]], filepath: str):
    """
    Save data to CSV file.

    Args:
        data: List of dictionaries
        filepath: Output file path
    """
    if len(data) == 0:
        print("No data to save")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = list(data[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {filepath}")


def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    data = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    return data


def save_pickle(data: Any, filepath: str):
    """
    Save data using pickle.

    Args:
        data: Data to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def save_experiment_results(results: Dict[str, Any], output_dir: str,
                           experiment_name: str):
    """
    Save experiment results in multiple formats.

    Args:
        results: Results dictionary
        output_dir: Output directory
        experiment_name: Name of the experiment
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(output_dir, f"{experiment_name}.json")
    save_json(results, json_path)

    # Save as pickle for full Python object preservation
    pickle_path = os.path.join(output_dir, f"{experiment_name}.pkl")
    save_pickle(results, pickle_path)

    # Save summary as CSV if results contain tabular data
    if 'summary' in results and isinstance(results['summary'], list):
        csv_path = os.path.join(output_dir, f"{experiment_name}_summary.csv")
        save_csv(results['summary'], csv_path)
