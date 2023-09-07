# Load a JSON dataset from local directory - We load training set only
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd

# Load Dataset
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# Slice and index
drug_dataset
drug_dataset['train'][0:5]

# View columns
drug_dataset['train'].column_names

# Check Dataset Shape
drug_dataset['train'].shape

# Filter rows
drug_dataset_filtered = drug_dataset.filter(lambda x: x['rating'] == 9.0)

# Subset Columns
drug_dataset_filtered['train'].select_columns(['rating', 'drugName'])

# Mutate Column
def convert_to_lowercase(example):
    if isinstance(example['condition'], str):
        new_col_dict = {"condition_lower": example["condition"].lower()}
    else:
        new_col_dict = {"condition_lower": example["condition"]}

    return new_col_dict

drug_dataset_filtered = drug_dataset_filtered.map(convert_to_lowercase)

# Rename Column
drug_dataset_filtered = drug_dataset_filtered.rename_columns({'usefulCount': 'useful_count', 'drugName': 'drug_name'})

# Sort Column
drug_dataset_sorted = drug_dataset.sort(["rating", "usefulCount"], reverse=[True, False])

# Make a train test split
drug_dataset_partitioned = drug_dataset['train'].train_test_split(train_size=0.8, seed=42)
## Rename the default "test" split to "validation". Popping brings out a data slice key
drug_dataset_partitioned["validation"] = drug_dataset_partitioned.pop("test")
## Add the "test" set to our `DatasetDict`
drug_dataset_partitioned["test"] = drug_dataset["test"]

# Convert to Pandas
drug_dataset_partitioned.set_format("pandas")
test_df = drug_dataset_partitioned['test'][:]

test_df_grped = test_df.groupby(by = ['drugName'], as_index = False).agg({'usefulCount': 'max'})

# Convert from Pandas to Dataset
test_df_grped_dataset = Dataset.from_pandas(test_df_grped)

drug_dataset_partitioned.reset_format()

# Save Data
from pathlib import Path

Path("saved_data/").mkdir(parents=True, exist_ok=True)

## Arrow format
drug_dataset_partitioned.save_to_disk("./saved_data/arrow/drug-data-partitioned")

## CSV format
for split, dataset in drug_dataset_partitioned.items():
    dataset.to_csv(f"./saved_data/CSV/drug-data-partitioned-{split}.csv")

## JSON format
for split, dataset in drug_dataset_partitioned.items():
    dataset.to_json(f"./saved_data/JSON/drug-data-partitioned-{split}.jsonl")

# Load Data
## Arrow format
drug_dataset_arrow_reloaded = load_from_disk("./saved_data/arrow/drug-data-partitioned")

## CSV format
data_files = {
    "train": "./saved_data/CSV/drug-data-partitioned-train.csv",
    "validation": "./saved_data/CSV/drug-data-partitioned-validation.csv",
    "test": "./saved_data/CSV/drug-data-partitioned-test.csv",
    }

drug_dataset_csv_reloaded = load_dataset("csv", data_files=data_files)

## JSON format
data_files = {
    "train": "./saved_data/JSON/drug-data-partitioned-train.jsonl",
    "validation": "./saved_data/JSON/drug-data-partitioned-validation.jsonl",
    "test": "./saved_data/JSON/drug-data-partitioned-test.jsonl",
    }

drug_dataset_json_reloaded = load_dataset("json", data_files=data_files)