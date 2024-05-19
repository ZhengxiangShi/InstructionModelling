import os
import pandas as pd
import argparse
import json
import numpy as np
import glob
import tqdm
import random
from length_analysis_dataset import encode_with_messages_format, check_length


def save_subset(directory, data_path, subset, alpha, target_count):
    # Format the filename to include alpha and the target count of the subset
    subset_path = os.path.join(directory, "{}_alpha{}_count{}_subset.jsonl".format(data_path, alpha, target_count))
    with open(subset_path, "w") as file:
        for example in subset:
            json.dump(example, file)
            file.write('\n')
            
def select_examples_matching_ratio(generation_results, alpha, target_count=3000):
    ratios = []
    for example in generation_results:
        instruction_length, output_length = encode_with_messages_format(example)
        if output_length > 0:  # To avoid division by zero
            ratio = instruction_length / output_length
        else:
            ratio = 0
        ratios.append((example, ratio))
    
    # Sort by difference from alpha, aiming to get the closest matches
    ratios.sort(key=lambda x: abs(x[1] - alpha))

    # Select the first `target_count` examples with the closest ratios
    selected_examples = [example for example, ratio in ratios[:target_count]]
    return selected_examples


def check_length(directory, data_path, alpha, target_count):
    dataset_length = {}

    data_file = os.path.join(directory, "{}.jsonl".format(data_path))

    with open(data_file, "r") as f_data:
        generation_results = [json.loads(line) for line in f_data]
        random.shuffle(generation_results)

        # Select subset where average ratio is closest to alpha
        selected_subset = select_examples_matching_ratio(generation_results, alpha, target_count)

        # Save the selected subset
        save_subset(directory, data_path, selected_subset, alpha, target_count)

        # Optionally calculate the average ratios in the subset if needed for verification
        subset_ratios = [instruction_length / output_length if output_length > 0 else 0 
                            for example in selected_subset 
                            for instruction_length, output_length in [encode_with_messages_format(example)]]
        average_ratio = np.mean(subset_ratios)
        print(f'Average ratio: {average_ratio}')
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dir", 
        type=str, 
        default="data"
    )
    arg_parser.add_argument(
        "--data", 
        type=str, 
        default="processed/tulu_v2/tulu_v2_data"
    )
    arg_parser.add_argument(
        "--alpha", 
        type=float, 
        default=1.0
    )
    arg_parser.add_argument(
        "--target_count", 
        type=int, 
        default=3000,
    )
    args = arg_parser.parse_args()

    check_length(args.dir, args.data, args.alpha, args.target_count)