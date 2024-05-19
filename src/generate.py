import ast
import sys
import os
import json
import argparse
import logging
import random
import torch
import datasets
import vllm


def get_instruction(example):
    messages = example["messages"]

    message = messages[0]
    if message["role"] == "user":
        return message["content"]
    else:
        raise ValueError(
            "Llama2 chat template only supports 'user' role. Invalid role: {}.".format(message["role"])
        )


def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    # Check whether the output has been generated before
    if os.path.exists(os.path.join(args.save_dir, "output.json")):
        print("Output already exists")
    else:
        data_file = os.path.join(args.data_dir, "{}.jsonl".format(args.data_path))
        with open(data_file, "r") as f_data:
            dataset = [json.loads(line) for line in f_data]
            # dataset = dataset[:10]

        prompts = []
        for example in dataset:
            prompt = get_instruction(example)

            messages = [{"role": "user", "content": prompt}]
            prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
            prompts.append(prompt)

        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=args.max_new_tokens,
            )
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
            )


        model_results = []
        with open(os.path.join(args.save_dir, "output.json"), "w") as fout:
            for example, output in zip(dataset, outputs):
                if "messages" in example:
                    example.pop("messages")
                example["output"] = output
                example["generator"] = f"{model_name}-greedy-long"
                fout.write(json.dumps(example) + "\n")
                model_results.append(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results_overfitting")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="processed/tulu_v2/lima_subset/lima_data",
        help="The path to the data file."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=20, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    args = parser.parse_args()

    main(args)