"""
This script converts the datasets from the AlPaGAsus paper into the instruction-following format.
```
[
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Variable x is defined as \u201c4x + 2y = 10\u201d. Find the value of x.",
        "input": "",
        "output": "The value of x is 2. To find the value, simplify the equation by subtracting 2y from both sides, giving 4x = 10; dividing both sides by 4, giving x = 2/4, which is equal to 2."
    },
]
```
into the following format:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
"""

import json
import random
import re
import os
import pandas as pd
import argparse
from instruction_encode_templates import encode_instruction_example, encode_few_shot_example


def convert_alpagasus_alpaca_format(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(data_dir, "claude_t45.json"), "r") as f:
        examples = json.load(f)

    output_path = os.path.join(output_dir, "alpagasus_claude_t45_alpaca.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "alpagasus_claude_t45_alpaca",
                "id": f"alpagasus_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_alpagasus_dolly_format(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_file, size in zip(["chatgpt_9k.json", "dolly_3k.json"], ["9k", "3k"]):
        with open(os.path.join(data_dir, data_file), "r") as f:
            examples = json.load(f)
        output_path = os.path.join(output_dir, "alpagasus_{}_dolly.jsonl".format(size))
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                encoded_example = encode_instruction_example(
                    instruction=example["instruction"], 
                    input=example["input"], 
                    output=example["output"],
                    random_template=True,
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "alpagasus_{}_dolly".format(size),
                    "id": f"alpagasus_{size}_{idx}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default="./data/alpagasus"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/processed/alpagasus"
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)
    
    convert_alpagasus_alpaca_format(
        os.path.join(args.raw_data_dir, "alpaca"),
        args.output_dir,
    )

    convert_alpagasus_dolly_format(os.path.join(args.raw_data_dir, "dolly"), args.output_dir)    
    