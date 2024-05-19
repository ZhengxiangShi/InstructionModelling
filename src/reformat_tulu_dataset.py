""" 
This script is used to download the tulu dataset from the huggingface website.
Then we convert the downloaded dataset to processed instruction tuning dataset.
"""
import json
import random
import re
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from instruction_encode_templates import encode_instruction_example, encode_few_shot_example


def convert_tulu_format(data_path, output_dir, percentage=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tulu_dataset = []
    dataset_source_count = defaultdict(int)
    with open(data_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            example = json.loads(line)
            tulu_dataset.append(example)
            dataset_source_count[example["dataset"]] += 1

    # Calculate the distribution of languages
    total_count = len(tulu_dataset)
    sample_size = int(percentage * total_count)
    print("We sample {}% of the dataset".format(percentage * 100))
    print("This results in {} samples".format(sample_size))

    # Calculate the number of samples for each language based on its proportion
    samples_per_dataset = {lang: int((count / total_count) * sample_size) for lang, count in dataset_source_count.items()}
    discrepancy = sample_size - sum(samples_per_dataset.values())
    print("Total samples:", sample_size)
    print("According to the distribution of datasets, the number of samples for each dataset is:")
    for lang, count in samples_per_dataset.items():
        print(f"{lang}: {count}")

    # Sample the dataset for each language
    examples = []
    for example in tqdm(tulu_dataset):
        dataset_source = example['dataset']
        if samples_per_dataset[dataset_source] > 0:
            examples.append(example)
            samples_per_dataset[dataset_source] -= 1
        elif discrepancy > 0:
            examples.append(example)
            discrepancy -= 1
        else:
            continue

    output_path = os.path.join(output_dir, "tulu_dataset_{}.jsonl".format(str(percentage).replace(".", "")))
    with open(output_path, "w") as fout:
        for _, example in enumerate(examples):
            fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/processed/tulu_v2/tulu_v2_data.jsonl"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed/tulu_v2/"
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)

    for p in [0.1, 0.2, 0.5]:    
        convert_tulu_format(
            args.data_path,
            args.output_dir,
            percentage=p,
        )
    