#!/usr/bin/env python3
"""
get_missing_api.py - Find API calls in dataset that are not documented.

Compares unique API calls in MalBehavD-V1-dataset.csv against
the API documentation JSON to identify missing entries.
"""

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_API_DOCS = PROJECT_ROOT / "data" / "api_docs"


def main():
    # Load CSV and extract unique API calls
    csv_path = DATA_RAW / "MalBehavD-V1-dataset.csv"
    df = pd.read_csv(csv_path)

    # Get all API columns (skip sha256 & labels at index 0,1)
    api_columns = df.columns[2:]

    # Flatten all API calls
    all_api = df[api_columns].values.flatten()

    # Remove empty values and get unique sorted list
    unique_api = sorted(set(x for x in all_api if isinstance(x, str) and x.strip()))

    print(f"Total unique API calls in CSV: {len(unique_api)}")

    # Load JSON documentation
    json_path = DATA_API_DOCS / "api_docs.json"
    with open(json_path, "r", encoding="utf-8") as f:
        api_docs = json.load(f)

    nt_keys = set(api_docs.keys())

    # Compare CSV vs JSON
    not_in_json = [api for api in unique_api if api not in nt_keys]

    print("\nAPI calls in CSV not found in JSON documentation:")
    for api in not_in_json:
        print(f"  - {api}")

    print(f"\nTotal missing API calls: {len(not_in_json)}")

    # Save to file
    output_path = PROJECT_ROOT / "results" / "missing_api_calls.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("API calls in dataset not found in documentation:\n")
        f.write("=" * 60 + "\n\n")
        for api in not_in_json:
            f.write(f"{api}\n")
    
    print(f"\nSaved missing API list to: {output_path}")


if __name__ == "__main__":
    main()
