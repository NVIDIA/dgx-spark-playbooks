#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from unsloth import FastVisionModel

import re
import sys
import yaml
import shutil
import signal

from PIL import ImageFile
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def load_model_for_train(config):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config["model"]["model_id"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config["model"]["finetune_vision_layers"],
        finetune_language_layers=config["model"]["finetune_language_layers"],
        finetune_attention_modules=config["model"]["finetune_attention_modules"],
        finetune_mlp_modules=config["model"]["finetune_mlp_modules"],
        r=config["model"]["lora_config"]["rank"],
        lora_alpha=config["model"]["lora_config"]["alpha"],
        lora_dropout=config["model"]["lora_config"]["dropout"],
        bias="none",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def format_instruction(sample, label_dict):
    label = label_dict.int2str(sample["label"])
    if label == "nowildfire":
        answer = "No"
    else:
        answer = "Yes"

    # reasoning prompt
    prompt = "Identify if this region has been affected by a wildfire"
    prompt = (
        f"{prompt}. Also first provide your reasoning or working out"\
        f" on how you would go about identifying the presence of wildfire affected regions between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and (put a simple Yes or No here) {SOLUTION_END}"
    )

    # convert image format
    image = sample["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    # construct instruction prompt
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    return {"prompt": prompt, "image": sample["image"], "answer": answer}


def load_wildfire_dataset(config, tokenizer):
    # load dataset
    train_dataset = load_dataset(config["data"]["dataset_id"])["train"]

    # preprocess the dataset
    train_dataset = train_dataset.map(
        lambda sample: format_instruction(sample, train_dataset.features["label"]),
        num_proc=8)
    train_dataset = train_dataset.map(
        lambda sample: {
            "prompt": tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
        }, num_proc=8)

    return train_dataset


def format_reward_func(completions, **kwargs):
    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    scores = []
    for completion in completions:
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0

        # penalize excessive addCriterion predictions in qwen2.5-vl
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion)-len(removal))/len(completion) >= 0.5:
                score -= 1.0

        scores.append(score)

    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    q = prompts[0]
    print("----------------------------------")
    print(f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{completions[0]}")
    return [
        5.0 if len(r)==1 and a == r[0].replace('\n','') else 0.0
        for r, a in zip(responses, answer)
    ]


def start_train(config):

    # load dataset
    train_dataset = load_wildfire_dataset(config, tokenizer)

    # define training arguments
    training_args = GRPOConfig(
        learning_rate=config["hyperparameters"]["learning_rate"],
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=1,
        log_completions=False,
        per_device_train_batch_size=config["hyperparameters"]["batch_size"],
        gradient_accumulation_steps=1, 
        num_generations=config["hyperparameters"]["num_generations"],
        max_prompt_length=config["model"]["max_seq_length"],
        max_completion_length=config["model"]["max_seq_length"],
        max_steps=config["hyperparameters"]["steps"],
        save_steps=5,
        save_total_limit=2,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=config["hyperparameters"]["output_dir"],
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
    )

    # start training
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=[
            format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
    )
    trainer.train()

    handle_termination(None, None)


def handle_termination(signum, frame):
    latest_checkpoint = get_last_checkpoint(config["hyperparameters"]["output_dir"])
    if latest_checkpoint:
        if config["model"]["use_lora"]:
            print("Merging LoRA weights and saving the model")
            shutil.rmtree(latest_checkpoint)
            model.save_pretrained_merged(latest_checkpoint, tokenizer, save_method="merged_16bit")

    sys.exit(0)


signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)


if __name__ == "__main__":
    with open("src/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # load base model for finetuning
    model, tokenizer = load_model_for_train(config)

    start_train(config)
