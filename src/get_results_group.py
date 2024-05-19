""" 
This script is used to get the results from the output of the model.
We will compute an average of the results for each group and task. For each task, we report one metric.
"""
import os
import pandas as pd
import argparse
import json
import numpy as np
import glob

METRIC_DCT = {
    "sft_world_knowledge": {
        "name": "Language Understanding and Knowledge",
        "tasks": {
            "mmlu": ["acc,none", "acc_stderr,none"],
            "piqa": ["acc_norm,none", "acc_norm_stderr,none"],
            "openbookqa": ["acc_norm,none", "acc_norm_stderr,none"],
            "hellaswag": ["acc_norm,none", "acc_norm_stderr,none"],
            "lambada": ["acc,none", "acc_stderr,none"],
        },
    },
    "sft_multilinguality": {
        "name": "Multilinguality",
        "tasks": {
            "lambada_multilingual": ["acc,none", "acc_stderr,none"],
            "gpt3_translation_benchmarks": ["ter,none", "ter_stderr,none"],
        },
    },
    "sft_commonsense_reasoning": {
        "name": "Commonsense Reasoning",
        "tasks": {
            "wsc273": ["acc,none", "acc_stderr,none"],
            "winogrande": ["acc,none", "acc_stderr,none"],
            "ai2_arc": ["acc_norm,none", "acc_norm_stderr,none"],
            "coqa": ["f1,none", "f1_stderr,none"],
        },
    },
    "sft_symbolic_problem_solving": {
        "name": "Math and Coding Reasoning",
        "tasks": {
            "gsm8k_cot": ["exact_match,strict-match", "exact_match_stderr,strict-match"],
            "human_eval": None,
        },
    },
    "sft_bbh_cot_fewshot": {
        "name": "Few-shot Learning",
        "tasks": {
            "bbh_cot_fewshot": ["exact_match,get-answer", "exact_match_stderr,get-answer"],
        },
    },
    "sft_safety": {
        "name": "Safety and Helpfulness",
        "tasks": {
            "truthfulqa_mc2": ["acc,none", "acc_stderr,none"],
            "toxigen": ["acc,none", "acc_stderr,none"],
            "hendrycks_ethics": ["acc,none", "acc_stderr,none"],
        },
    },
}


def compute_model_performance(results_dct, human_eval_performance, human_eval_metric, base_results=None, base_average=None):
    """ 
    This function is used to compute the average of the results for each group and task.
    Store them in the pandas dataframe.
    """
    # Create a pandas dataframe to store the results
    results = pd.DataFrame(columns=["group", "task", "metric", "result", "stderr"])

    group_performance_store = []
    for _, sub_dct in METRIC_DCT.items():
        group = sub_dct["name"]
        average_group_performance = []
        for task, metric_list in sub_dct["tasks"].items():
            if task == "human_eval":
                metric_result = human_eval_performance
                stderr_result = 0
                metric_name = human_eval_metric
            else:
                metric_name, stderr_name = metric_list[0], metric_list[1]
                metric_result = results_dct[task][metric_name]
                stderr_result = results_dct[task][stderr_name]
                if metric_name == "ter,none":
                    metric_result = metric_result / 100
            if base_results is not None:
                base_metric_result = base_results[(base_results["task"] == task) & (base_results["group"] == group)]["result"].values[0]
            new_row = {
                "group": group,
                "task": task,
                "metric": metric_name,
                "result": metric_result,
                "stderr": stderr_result,
                "impr": 100 * (metric_result - base_metric_result) if base_results is not None else 0,
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
            average_group_performance.append(metric_result)
        group_performance_store.append((
            group,
            np.mean(average_group_performance),
            np.std(average_group_performance),
        ))
    # Add the average performance of the group to the dataframe
    text_output = []
    for group, mean, std in group_performance_store:
        if base_results is not None:
            base_metric_result = base_results[(base_results["task"] == "average") & (base_results["group"] == group)]["result"].values[0]
        impr = 100 * (mean - base_metric_result) if base_results is not None else 0
        new_row = {
            "group": group,
            "task": "average",
            "metric": "average",
            "result": mean,
            "stderr": std,
            "impr": impr,
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        # if impr > 0:
        #     text = r"{:.2f}".format(100*mean) + r"\ua{" + r"{:.2f}".format(impr) + "}"
        # else:
        #     text = r"{:.2f}".format(100*mean) + r"\da{" + r"{:.2f}".format(-impr) + "}"
        text = r"{:.2f}".format(100*mean)
        text_output.append(
            text
        )
    # Print the results in markdown format
    markdown_results = results.to_markdown(index=False)

    average_performance = np.mean([mean*100 for _, mean, _ in group_performance_store])
    if base_average is None:
        text_output = " & ".join(text_output) + " & {:.2f}".format(average_performance)
    else:
        if average_performance > base_average:
            text_output = " & ".join(text_output) + " & {:.2f}".format(average_performance) + r"\ua{" + r"{:.2f}".format(average_performance - base_average) + "}"
        else:
            text_output = " & ".join(text_output) + " & {:.2f}".format(average_performance) + r"\da{" + r"{:.2f}".format(base_average - average_performance) + "}"
    return markdown_results, results, text_output, average_performance


def get_human_eval_results(humaneval_path, model_name, metric_name="pass@1"):
    """ 
    This function is used to get the results for the Humaneval dataset.
    """
    best_metric = None
    for temperature in ["01", "07"]:
        with open(os.path.join(humaneval_path, "{}_t{}".format(model_name, temperature), "metrics.json".format(temperature)), "r") as f:
            results = json.load(f)[metric_name]
            if best_metric is None or results > best_metric:
                best_metric = results
    return best_metric


def compute_llm_eval_performance(path, model_name):
    try:
        with open(os.path.join(path, "results_alpaca_eval", model_name, "metrics.json"), "r") as f:
            results = json.load(f)
            alpacaeval_result_v2 = results["win_rate"]["{}-greedy-long".format(model_name)]
    except FileNotFoundError:
        alpacaeval_result_v2 = 0.00

    try:
        with open(os.path.join(path, "results_alpaca_eval", model_name + "_v1", "metrics.json"), "r") as f:
            results = json.load(f)
            if "epoch" in model_name:
                key_name = "{}-greedy-long".format(model_name[-7:])
            else:
                key_name = "{}-greedy-long".format(model_name)
            alpacaeval_result_v1 = results["win_rate"][key_name]
    except FileNotFoundError:
        alpacaeval_result_v1 = 0.00

    # Read the saved file
    try:
        mt_bench_df = pd.read_csv(os.path.join(path, "mt_bench_average_results.csv"))
        # Get the results for the dataframe based on the model name
        
        # Check whethe the model_name is in the dataframe
        if model_name not in mt_bench_df["model"].values:
            mt_bench_result = 0.00
        else:
            mt_bench_result = mt_bench_df[mt_bench_df["model"] == model_name]["score"].values[0]
    except FileNotFoundError:
        mt_bench_result = 0.00

    return mt_bench_result, alpacaeval_result_v1, alpacaeval_result_v2


def get_results(path, humaneval_path, llm_eval_path, human_eval_metric="pass@1"):
    """ 
    This function is used to get the results from the output of the model.
    Based on METRIC_DCT, we will get the results from the output of the model.
    Steps:
    1. Get the results for each task, according to the corresponding metric in sub-dct.
    2. Compute the average of the results for each group and task.
    """
    with open(os.path.join(path, "overall.md"), "w") as f:
        all_files = glob.glob(path + "/**/results.json", recursive=True)
        for idx in range(len(all_files)):
            if all_files[idx].split("/")[-2] == "Llama-2-7b-hf":
                base_file_index = idx

        base_file = all_files.pop(base_file_index)
        model_name = base_file.split("/")[-2]
        f.write("## {}\n".format(model_name))
        
        # Get the results for traditional NLP benchmarks
        with open(base_file, "r") as ff:
            results_dct = json.load(ff)
            
        # Get the results for the Humaneval dataset
        try:
            human_eval_performance = get_human_eval_results(
                humaneval_path,
                model_name,
                metric_name=human_eval_metric,
            )
        except FileNotFoundError:
            human_eval_performance = 0

        markdown_results, base_results, text_output, base_average = compute_model_performance(
            results_dct["results"],
            human_eval_performance,
            human_eval_metric,
            None,
        )

        mt_bench_result, alpacaeval_result_v1, alpacaeval_result_v2 = compute_llm_eval_performance(
            llm_eval_path,
            model_name
        )
        base_average_llm_eval = (mt_bench_result + alpacaeval_result_v1 + alpacaeval_result_v2) / 3
        text_output += " & {:.2f}".format(mt_bench_result) + " & {:.2f}".format(alpacaeval_result_v1) + " & {:.2f}".format(alpacaeval_result_v2) + " & {:.2f}".format(base_average_llm_eval)

        print(markdown_results)
        # Write the results to the file
        f.write(markdown_results)
        f.write("\n")
        f.write(text_output)
        f.write("\n")
        f.write("The average performance of the model is {:.2f}".format(base_average) + "\n")
        f.write("\n\n")

        for file in all_files:
            model_name = file.split("/")[-2]
            f.write("## {}\n".format(model_name))
            
            # Get the results for traditional NLP benchmarks
            with open(file, "r") as ff:
                results_dct = json.load(ff)
                
            # Get the results for the Humaneval dataset
            try:
                human_eval_performance = get_human_eval_results(
                    humaneval_path,
                    model_name,
                    metric_name=human_eval_metric,
                )
            except FileNotFoundError:
                human_eval_performance = 0

            markdown_results, _, text_output, model_average = compute_model_performance(
                results_dct["results"],
                human_eval_performance,
                human_eval_metric,
                base_results,
                base_average,
            )
            
            trained_mt_bench_result, trained_alpacaeval_result_v1, trained_alpacaeval_result_v2 = compute_llm_eval_performance(
                llm_eval_path,
                model_name
            )
            
            llm_eval_output = ""
            for trained_result, base_result in zip([trained_mt_bench_result, trained_alpacaeval_result_v1, trained_alpacaeval_result_v2], [mt_bench_result, alpacaeval_result_v1, alpacaeval_result_v2]):
                if trained_result > base_result:
                    llm_eval_output += " & {:.2f}".format(trained_result) + r"\ua{" + r"{:.2f}".format(trained_result - base_result) + "}"
                else:
                    llm_eval_output += " & {:.2f}".format(trained_result) + r"\da{" + r"{:.2f}".format(base_result - trained_result) + "}"
            text_output += llm_eval_output
            # average_llm_eval = (mt_bench_result + alpacaeval_result_v1 + alpacaeval_result_v2) / 3
            # if average_llm_eval > base_average_llm_eval:
            #     text_output += " & {:.2f}".format(mt_bench_result) + " & {:.2f}".format(alpacaeval_result_v1) + " & {:.2f}".format(alpacaeval_result_v2) + " & {:.2f}".format(average_llm_eval) + r"\ua{" + r"{:.2f}".format(average_llm_eval - base_average_llm_eval) + "}"
            # else:
            #     text_output += " & {:.2f}".format(mt_bench_result) + " & {:.2f}".format(alpacaeval_result_v1) + " & {:.2f}".format(alpacaeval_result_v2) + " & {:.2f}".format(average_llm_eval) + r"\da{" + r"{:.2f}".format(base_average_llm_eval - average_llm_eval) + "}"
            # text_output += " & {:.2f}".format(mt_bench_result) + " & {:.2f}".format(alpacaeval_result_v1) + " & {:.2f}".format(alpacaeval_result_v2) + " & {:.2f}".format(average_llm_eval)

            print(markdown_results)
            # Write the results to the file
            f.write(markdown_results)
            f.write("\n")
            f.write(text_output)
            f.write("\n")
            f.write(llm_eval_output)
            f.write("\n")
            f.write("The average performance of the model is {:.2f}".format(model_average) + "\n")
            f.write("\n\n")



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dir", 
        type=str, 
        default="output"
    )
    arg_parser.add_argument(
        "--humaneval_dir", 
        type=str, 
        default="results_humaneval"
    )
    arg_parser.add_argument(
        "--llm_eval_dir", 
        type=str, 
        default="."
    )
    args = arg_parser.parse_args()

    get_results(args.dir, args.humaneval_dir, args.llm_eval_dir)