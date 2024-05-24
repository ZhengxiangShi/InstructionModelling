# Instruction Tuning With Loss Over Instructions
This repository provides the code for our paper titled **[Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394)**, making the integration of our code contributions into other projects more accessible.

<div align="center">

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2405.14394)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

<p align="center">
  <img src="asset/insight.png" width="1000"></a>
  <br />
  <em>Our study further identifies key factors influencing the effectiveness of Instruction Modelling: (1) The ratio between instruction length and output length. (Left Figure). (2) The number of training examples. (Right Figure).</em>
</p>


## Quick Links
- [Instruction Tuning With Loss Over Instructions](#instruction-tuning-with-loss-over-instructions)
  - [Quick Links](#quick-links)
  - [Overview](#overview)
  - [1. Requirements and Installation](#1-requirements-and-installation)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Reproducing Analysis](#4-reproducing-analysis)
  - [Bugs or questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Overview
You can reproduce the experiments of our paper [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394).

> **Abstract**
> Instruction tuning plays a crucial role in shaping the outputs of language models (LMs) to desired styles. In this work, we propose a simple yet effective method, Instruction Modelling (IM), which trains LMs by applying a loss function to the instruction and prompt part rather than solely to the output part. Through experiments across 21 diverse benchmarks, we show that, in many scenarios, IM can effectively improve the LM performance on both NLP tasks (e.g., MMLU, TruthfulQA, and HumanEval) and open-ended generation benchmarks (e.g., MT-Bench and AlpacaEval). Remarkably, in the most advantageous case, IM boosts model performance on AlpacaEval 1.0 by over 100%. We identify two key factors influencing the effectiveness of IM: (1) The ratio between instruction length and output length in the training data; and (2) The number of training examples. We observe that IM is especially beneficial when trained on datasets with lengthy instructions paired with brief outputs, or under the Superficial Alignment Hypothesis (SAH) where a small amount of training examples are used for instruction tuning. Further analysis substantiates our hypothesis that the improvement can be attributed to reduced overfitting to instruction tuning datasets. Our work provides practical guidance for instruction tuning LMs, especially in low-resource scenarios. 
> 

## 1. Requirements and Installation
To install the required packages for our baseline approaches (semi-supervised approaches), you can run the following command.
```sh
conda create -n sft python=3.10
conda activate sft
pip install -r requirements.txt
```

For the training data, we have provided the processed data in the `data` directory for 7 instruction tuning datasets. You can download other data from the following links:
```sh 
sh prepare_train_data.sh
```
In addition, we download the less data from the the [Princeton NLP Less Data](https://huggingface.co/datasets/princeton-nlp/less_data/blob/main/less-data.zip).

To download the data for the Alpagasus dataset, you can run the following command.
```sh
sh prepare_alpagasus_data.sh
```

## 2. Training
Here we provide the instructions for training the models for the standard instruction tuning, instruction modelling (ours), and the baseline models.

To train the instruction tuning model, you can run the following command.
```sh
export CUDA_VISIBLE_DEVICES=0,1
MODEL_SIZE=7b
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
EPOCH=2
MAX_LENGTH=2048 
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

DATA_NAME_LIST=(
    lima_data \
    alpagasus_3k_dolly \
    alpagasus_9k_dolly \
    alpagasus_claude_t45_alpaca \
    tydiqa \
    mmlu_chat \
    bbh_icl \
)
DATASET_PATH_LIST=(
    lima_data \
    alpagasus_3k_dolly \
    alpagasus_9k_dolly \
    alpagasus_claude_t45_alpaca \
    tydiqa_adam_sim_trainp0.05_seed3_p0.05 \
    mmlu-chat_adam_sim_trainp0.05_seed3_p0.05 \
    bbh-icl_adam_sim_trainp0.05_seed3_p0.05 \
)
for i in "${!DATA_NAME_LIST[@]}"; do
    DATA_NAME=${DATA_NAME_LIST[i]}
    DATASET_PATH=${DATASET_PATH_LIST[i]}
    for LR in 2e-5; do
        DATA_PATH=data/${DATASET_PATH}.jsonl
        OUTPUT_DIR=model/${DATA_NAME}_llama2_${MODEL_SIZE}_bs${TOTAL_BATCH_SIZE}_lr${LR}_ml${MAX_LENGTH}_ep${EPOCH}_bf16
        printf '%q\n%q\n%q\n%q\n' "$DATA_NAME" "$DATASET_PATH" "$DATA_PATH" "$OUTPUT_DIR"

        accelerate launch \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes $NUM_GPUS \
            --use_deepspeed \
            --main_process_port 29521 \
            --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
            src/finetune.py \
            --model_name_or_path meta-llama/Llama-2-${MODEL_SIZE}-hf \
            --use_flash_attn \
            --tokenizer_name meta-llama/Llama-2-${MODEL_SIZE}-hf \
            --use_slow_tokenizer \
            --train_file ${DATA_PATH} \
            --max_seq_length ${MAX_LENGTH} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate ${LR} \
            --lr_scheduler_type linear \
            --warmup_ratio 0.03 \
            --weight_decay 0. \
            --checkpointing_steps epoch \
            --num_train_epochs ${EPOCH} \
            --output_dir ${OUTPUT_DIR} \
            --with_tracking \
            --report_to tensorboard \
            --logging_steps 1;
    done;
done
```

To train the instruction modelling model, you can run the following command. This is our proposed method.
```sh
for i in "${!DATA_NAME_LIST[@]}"; do
    DATA_NAME=${DATA_NAME_LIST[i]}
    DATASET_PATH=${DATASET_PATH_LIST[i]}
    for LR in 2e-5; do
        DATA_PATH=data/${DATASET_PATH}.jsonl
        OUTPUT_DIR=model/${DATA_NAME}_llama2_${MODEL_SIZE}_bs${TOTAL_BATCH_SIZE}_lr${LR}_ml${MAX_LENGTH}_ep${EPOCH}_bf16_im
        printf '%q\n%q\n%q\n%q\n' "$DATA_NAME" "$DATASET_PATH" "$DATA_PATH" "$OUTPUT_DIR"

        accelerate launch \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes $NUM_GPUS \
            --use_deepspeed \
            --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
            src/finetune.py \
            --model_name_or_path meta-llama/Llama-2-${MODEL_SIZE}-hf \
            --use_flash_attn \
            --tokenizer_name meta-llama/Llama-2-${MODEL_SIZE}-hf \
            --use_slow_tokenizer \
            --train_file ${DATA_PATH} \
            --max_seq_length ${MAX_LENGTH} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate ${LR} \
            --lr_scheduler_type linear \
            --warmup_ratio 0.03 \
            --weight_decay 0. \
            --checkpointing_steps epoch \
            --num_train_epochs ${EPOCH} \
            --output_dir ${OUTPUT_DIR} \
            --with_tracking \
            --report_to tensorboard \
            --logging_steps 1 \
            --use_lm_loss;
    done;
done
```

To train the baseline models (NefTune), you can run the following command.
```sh
NEFTUNE_ALPHA=5

for i in "${!DATA_NAME_LIST[@]}"; do
    DATA_NAME=${DATA_NAME_LIST[i]}
    DATASET_PATH=${DATASET_PATH_LIST[i]}
    for LR in 2e-5; do
        DATA_PATH=data/${DATASET_PATH}.jsonl
        OUTPUT_DIR=model/${DATA_NAME}_llama2_${MODEL_SIZE}_bs${TOTAL_BATCH_SIZE}_lr${LR}_ml${MAX_LENGTH}_ep${EPOCH}_bf16_alpha${NEFTUNE_ALPHA}
        printf '%q\n%q\n%q\n%q\n' "$DATA_NAME" "$DATASET_PATH" "$DATA_PATH" "$OUTPUT_DIR"

        accelerate launch \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes $NUM_GPUS \
            --use_deepspeed \
            --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
            src/finetune.py \
            --model_name_or_path meta-llama/Llama-2-${MODEL_SIZE}-hf \
            --use_flash_attn \
            --tokenizer_name meta-llama/Llama-2-${MODEL_SIZE}-hf \
            --use_slow_tokenizer \
            --train_file ${DATA_PATH} \
            --max_seq_length ${MAX_LENGTH} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate ${LR} \
            --lr_scheduler_type linear \
            --warmup_ratio 0.03 \
            --weight_decay 0. \
            --checkpointing_steps epoch \
            --num_train_epochs ${EPOCH} \
            --output_dir ${OUTPUT_DIR} \
            --with_tracking \
            --report_to tensorboard \
            --logging_steps 1 \
            --neftune_alpha ${NEFTUNE_ALPHA};
    done;
done
```

## 3. Evaluation
Here we provide the instructions for evaluating the models for the standard instruction tuning, instruction modelling (ours), and the baseline models.
We perform the evaluation using the open-source repository [FastChat](https://github.com/lm-sys/FastChat), [LLM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness), [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). Please refer to the respective repositories for more details. Please install the required packages for the evaluation.

To evaluate the model on traditional NLP tasks, you can run the following command.
```sh
CUDA_VISIBLE_DEVICES=0,1 
MODELS_0=(
    mmlu_chat_llama2_13b_bs128_lr2e-5_ml1024_ep2_bf16_im
)
(
    for model in ${MODELS_0}; do
        echo "Evaluating $model"
        MODEL_PATH=${BASE_PATH}/model/${model}
        echo ${MODEL_PATH}

        accelerate launch --mixed_precision bf16 --multi_gpu -m lm_eval --model hf \
            --model_args pretrained=${MODEL_PATH},max_length=${MAX_LENGTH} \
            --tasks sft_eval \
            --batch_size auto \
            --write_out \
            --show_config \
            --output_path output/${model} \
            --log_samples

        # CODEX: Evaluating using temperature 0.1 to get the pass@1 score
        python -m eval.codex_humaneval.run_eval \
            --data_file ${BASE_PATH}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir results_humaneval/${model}_t01 \
            --model ${MODEL_PATH} \
            --tokenizer ${MODEL_PATH} \
            --use_vllm

        # CODEX: Evaluating using temperature 0.8 to get the pass@10 score
        python -m eval.codex_humaneval.run_eval \
            --data_file ${BASE_PATH}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.7 \
            --save_dir results_humaneval/${model}_t07 \
            --model ${MODEL_PATH} \
            --tokenizer ${MODEL_PATH} \
            --use_vllm;
    done
)
```

To evaluate the model on the MT-Bench dataset, you can run the following command.
```sh
MODELS=mmlu_chat_llama2_13b_bs128_lr2e-5_ml1024_ep2_bf16_im
cd FastChat/fastchat/llm_judge

for model in $MODELS; do
    echo "Evaluating $model"

    echo "Firstly, Generate model answers to MT-bench questions"
    python gen_model_answer.py --model-path ${MODEL_PATH}/${model} --model-id ${model}

    echo "≈, Evaluate model answers using OpenAI API"
    python gen_judgment.py --model-list ${model} --parallel 2;
done

# To show the results
cd FastChat/fastchat/llm_judge
python show_result.py
python show_result.py --model-list model_name1 model_name2 # to show the results of the specified models
cd ../../../
```

To evaluate the model on the AlpacaEval dataset, you can run the following command.
```sh
MODELS=mmlu_chat_llama2_13b_bs128_lr2e-5_ml1024_ep2_bf16_im
export IS_ALPACA_EVAL_2=False
for model in $MODELS; do
    CUDA_VISIBLE_DEVICES=0 python -m eval.alpaca_farm.run_eval \
        --model_name_or_path  ${BASE_PATH}/${model} \
        --save_dir results_alpaca_eval/${model} \
        --eval_batch_size 20 \
        --use_vllm \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format;
done
```
Here you can set the `IS_ALPACA_EVAL_2` to `True` to evaluate the model on the AlpacaEval-2 dataset. If you just want to perform the generation without performing the evaluation, you can use the argument `--no_evaluate_with_llm`.

## 4. Reproducing Analysis
To reproduce the analysis of the paper, you can run the following command.

To compute the train or test loss of the model, you can run the following command.
```sh
MODEL_NMAES="lima_data_llama2_7b_bs128_lr2e-5_ml2048_ep2_bf16"
DATA_NAME_LIST=(
    lima_data \
    tulu_dataset_01 \
)
DATASET_PATH_LIST=(
    lima_data \
    tulu_dataset_01 \
)
for i in "${!DATA_NAME_LIST[@]}"; do
    DATA_NAME=${DATA_NAME_LIST[i]}
    DATASET_PATH=${DATASET_PATH_LIST[i]}
    DATA_PATH=data/${DATASET_PATH}.jsonl
    for model in $MODEL_NMAES; do
        accelerate launch \
            --main_process_port 29399 \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes $NUM_GPUS \
            --use_deepspeed \
            --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
            open_instruct/compute_loss.py \
            --model_name_or_path ${BASE_PATH}/${model} \
            --use_flash_attn \
            --tokenizer_name ${BASE_PATH}/${model} \
            --use_slow_tokenizer \
            --eval_file ${DATA_PATH} \
            --max_seq_length ${MAX_LENGTH} \
            --preprocessing_num_workers 16 \
            --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
            --output_dir output_loss/${model}_${DATA_NAME};
    done;
done
```

## Bugs or questions?
If you have any questions regarding the code or the paper, please feel free to reach out to Authors at `zhengxiang.shi.19@ucl.ac.uk`.  If you experience any difficulties while using the code or need to report a bug, feel free to open an issue. We kindly ask that you provide detailed information about the problem to help us provide effective support.

## Citation
```
@article{shi2024instruction,
title={Instruction Tuning With Loss Over Instructions},
author={Zhengyan Shi and Adam X. Yang and Bin Wu and Laurence Aitchison and Emine Yilmaz and Aldo Lipani},
booktitle={ArXiv},
year={2024},
url={https://arxiv.org/abs/2405.14394},
}
```

## Acknowledgements
We would like to thank the authors of the following repositories for providing the codebase:
- [FastChat](https://github.com/lm-sys/FastChat)
- [LLM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [open-instruct](https://github.com/allenai/open-instruct)
