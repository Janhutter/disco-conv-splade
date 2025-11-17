"""Create first-entry-only files from rewrite parquet files.

This script extracts just the first entry (current question) from each row
of the rewrite parquet files for multiple datasets.

For each dataset, it:
1. Reads the rewrites_mistral_test.parquet file
2. Extracts the 'text' field which contains the current question
3. Writes a TSV file with format: row_id \t current_question

Datasets processed:
- cast20_rewrites
- cast22_rewrites
- ikat23_rewrites
- qrecc_rewrites
- topiocqa_rewrites
"""

from pathlib import Path
import pandas as pd

# Define all datasets to process
DATASETS = [
    'cast20_rewrites',
    'cast22_rewrites', 
    'ikat23_rewrites',
    'qrecc_rewrites',
    'topiocqa_rewrites'
]

def create_first_entry_file(dataset_name: str):
    """Create first-entry-only TSV file for a dataset.
    
    Args:
        dataset_name: Name of the dataset directory (e.g., 'cast20_rewrites')
    """
    # Input parquet file
    parquet_path = Path(f'DATA/{dataset_name}/rewrites_mistral_test.parquet')
    
    if not parquet_path.exists():
        print(f"Warning: {parquet_path} not found, skipping...")
        return
    
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Output TSV with only first entries
    output_path = Path(f'DATA/{dataset_name}/rewrites_mistral_test_last.tsv')
    
    # Write TSV file with row_id and first entry (current question)
    with output_path.open('w') as f:
        for idx, row in df.iterrows():
            # Use the dataframe index as row_id
            # Text field contains the current question
            text = str(row['text']).strip()
            f.write(f"{idx}\t{text}\n")
    
    print(f"Created {output_path} ({len(df)} entries)")

def main():
    print("Creating first-entry-only files for all datasets...")
    print()
    
    for dataset in DATASETS:
        create_first_entry_file(dataset)
    
    print()
    print("Done!")


def create_topiocqa_human():
    input_path = Path('DATA/topiocqa_rewrites/rewrites_human.json')
    output_path = Path('DATA/topiocqa_rewrites/rewrites_t5_test_last.tsv')

    if not input_path.exists():
        print(f"Warning: {input_path} not found, skipping...")
        return

    import json
    with input_path.open('r') as f:
        data = json.load(f)

    # take question from each entry and write to tsv
    with output_path.open('w') as f:
        for idx, entry in enumerate(data):
            question = str(entry['question']).strip()
            f.write(f"{idx}\t{question}\n")
    print(f"Created {output_path} ({len(data)} entries)")

if __name__ == '__main__':
    main()
    create_topiocqa_human()