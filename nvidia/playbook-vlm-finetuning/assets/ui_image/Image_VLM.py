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

import os
import re
import json
import yaml
import glob
import time
import base64
import random
import requests
import subprocess

import pandas as pd
import streamlit as st
from transformers.trainer_utils import get_last_checkpoint


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def load_config():
    config_key = "config"
    if getattr(st.session_state, config_key, None) is None:
        with open("src/image_vlm_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        setattr(st.session_state, config_key, config)
    else:
        config = getattr(st.session_state, config_key)

    return config


@st.cache_resource
def start_vllm_server(model_id, model_type, max_seq_length, port):
    # get pwd
    return subprocess.Popen([
        "docker", "run",
        "--rm",
        "--gpus=all",
        "--ipc=host",
        "--net=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-v", f"{os.environ.get('HOST_HOME')}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{os.environ.get('HOST_PWD')}/ui_image/saved_model:/workspace/saved_model",
        "nvcr.io/nvidia/vllm:25.09-py3",
        "vllm", "serve",
        model_id,
        "--port", str(port),
        "--served-model-name", model_type,
        "--max-model-len", str(max_seq_length),
        "--gpu-memory-utilization", "0.45",
        "--async-scheduling",
        "--enable_prefix_caching"
    ])


def check_vllm_health(model_type, port):
    try :
        output = json.loads(subprocess.check_output(
            ["curl", "-s", f"http://localhost:{port}/v1/models"],
            text=True
        ))

        return output["data"][0]["id"] == model_type
    except:
        return False


def invoke_vllm_server(model_type, prompt, image, port):
    with open(image, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": model_type,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
    })

    return requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=payload
    ).json()["choices"][0]["message"]["content"]


def initialize_state(config):
    st.session_state["mode"] = st.session_state.get("mode", "inference")

    st.session_state["base"] = st.session_state.get("base", {})
    st.session_state["finetuned"] = st.session_state.get("finetuned", {})

    st.session_state["base"]["port"] = st.session_state["base"].get("port", "8000")
    st.session_state["finetuned"]["port"] = st.session_state["finetuned"].get("port", "8001")

    if st.session_state["mode"] == "inference":
        st.session_state["base"]["process"] = start_vllm_server(
            config["model_id"], "base", config["max_seq_length"], st.session_state["base"]["port"])
        finetuned_model_path = get_last_checkpoint(config["finetuned_model_id"])
        if finetuned_model_path is not None:
            st.session_state["finetuned"]["process"] = start_vllm_server(
                finetuned_model_path, "finetuned", config["max_seq_length"], st.session_state["finetuned"]["port"])

        if not check_vllm_health("base", st.session_state["base"]["port"]):
            with st.spinner("Loading vLLM server for base model..."):
                while not check_vllm_health("base", st.session_state["base"]["port"]):
                    time.sleep(1)
            st.toast("Base model loaded", icon="✅", duration="short")

        if finetuned_model_path is not None:
            if not check_vllm_health("finetuned", st.session_state["finetuned"]["port"]):
                with st.spinner("Loading vLLM server for finetuned model..."):
                    while not check_vllm_health("finetuned", st.session_state["finetuned"]["port"]):
                        time.sleep(1)
                st.toast("Finetuned model loaded", icon="✅", duration="short")

    st.session_state["current_image"] = st.session_state.get("current_image", glob.glob("assets/image_vlm/images/*/*")[-1])
    st.session_state["train_process"] = st.session_state.get("train_process", None)


def main():
    # set page ui
    st.title("Image VLM Finetuning")
    st.caption("A DGX Spark showcase for on-device VLM finetuning")
    # st.page_link("https://github.com/your-username/your-repo", label="GitHub", icon=":material/github:")

    # load css
    with open("src/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # load resources
    config = load_config()
    initialize_state(config["inference"])

    # train section
    st.markdown("---")
    train_section()

    # inference Section
    st.markdown("---")
    inference_section()


def train_section():
    st.header("GRPO Training")

    column_1, column_2, column_3 = st.columns(3, gap="large")
    with column_1:
        finetuning_method = st.selectbox(
        "Finetuning Method:",
            ["LoRA", "Full Fine-tuning"],
        )

    # update lora config
    if finetuning_method == "LoRA":
        lora_config = st.session_state["config"]["train"]["model"]["lora_config"]

        with column_2:
            lora_rank = st.slider(
                "LoRA Rank", 
                min_value=8,
                max_value=64, 
                value=lora_config["rank"], 
                step=8,
            )

        with column_3:
            lora_alpha = st.slider(
                "LoRA Alpha", 
                min_value=8,
                max_value=64, 
                value=lora_config["alpha"], 
                step=8,
            )

        st.session_state["config"]["train"]["model"]["lora_config"].update({
            'rank': lora_rank,
            'alpha': lora_alpha,
        })

    # update model config based on selection
    st.session_state["config"]["train"]["model"]["use_lora"] = finetuning_method == "LoRA"

    # update train config
    st.write("")
    column_1, column_2, column_3, column_4 = st.columns(4, gap="large")
    with column_1:
        finetune_vision_layers = st.toggle(
            "Finetune Vision Layers",
            value=st.session_state["config"]["train"]["model"]["finetune_vision_layers"])

    with column_2:
        finetune_language_layers = st.toggle(
            "Finetune Language Layers",
            value=st.session_state["config"]["train"]["model"]["finetune_language_layers"])

    with column_3:
        finetune_attention_modules = st.toggle(
            "Finetune Attention Modules",
            value=st.session_state["config"]["train"]["model"]["finetune_attention_modules"])

    with column_4:
        finetune_mlp_modules = st.toggle(
            "Finetune MLP Modules",
            value=st.session_state["config"]["train"]["model"]["finetune_mlp_modules"])

    st.write("")
    column_1, column_2, column_3, column_4 = st.columns(4, gap="large")
    with column_1:
        steps = st.slider(
            "Steps", 
            min_value=1,
            max_value=1000, 
            value=st.session_state["config"]["train"]["hyperparameters"]["steps"])

    with column_2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[1, 2, 4, 8, 16],
            value=st.session_state["config"]["train"]["hyperparameters"]["batch_size"])

    with column_3:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=float(st.session_state["config"]["train"]["hyperparameters"]["learning_rate"]),
            format="%.2e")

    with column_4:
        optimizer = st.selectbox(
            "Optimizer",
            options=["adamw_torch", "adafactor"])

    st.session_state["config"]["train"]["hyperparameters"].update({
        'steps': steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': optimizer,
    })

    st.session_state["config"]["train"]["model"].update({
        'finetune_vision_layers': finetune_vision_layers,
        'finetune_language_layers': finetune_language_layers,
        'finetune_attention_modules': finetune_attention_modules,
        'finetune_mlp_modules': finetune_mlp_modules,
    })

    st.write("")
    column_1, column_2, column_3, column_4 = st.columns(4, gap="large")
    with column_1:
        enable_grpo = st.toggle(
            "Enable GRPO",
            value=st.session_state["config"]["train"]["hyperparameters"]["enable_grpo"],
            disabled=True)

    with column_2:
        format_reward = st.number_input(
            "Reward for reasoning format",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state["config"]["train"]["hyperparameters"]["format_reward"]),
            format="%.2f")

    with column_3:
        correctness_reward = st.number_input(
            "Reward for correct response",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state["config"]["train"]["hyperparameters"]["correctness_reward"]),
            format="%.2f")

    with column_4:
        num_generations = st.number_input(
            "Number of generations",
            min_value=1,
            max_value=16,
            value=st.session_state["config"]["train"]["hyperparameters"]["num_generations"])

    # Training control
    st.write("")
    column_1, column_2, column_3 = st.columns([4, 4, 1])

    with column_1:
        button_type = "secondary" if st.session_state["train_process"] else "primary"
        if st.button("▶️ Start Finetuning", type=button_type, width="stretch", disabled=bool(st.session_state["train_process"])):
            if st.session_state["train_process"] is None:
                st.session_state["base"]["process"].terminate()
                st.session_state["base"]["process"].wait()
                st.session_state["base"]["process"] = None
                if "finetuned" in st.session_state and "process" in st.session_state["finetuned"]:
                    st.session_state["finetuned"]["process"].terminate()
                    st.session_state["finetuned"]["process"].wait()
                    st.session_state["finetuned"]["process"] = None
                st.session_state["mode"] = "train"
                st.cache_resource.clear()

                # store config
                with open("src/train.yaml", "w") as f:
                    yaml.dump(st.session_state["config"]["train"], f, default_flow_style=False)

                # start training
                with open("/tmp/logs.txt", "w") as f:
                    st.session_state["train_process"] = subprocess.Popen(
                        ["python", "-u", "src/train_image_vlm.py"],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                st.toast("Training started", icon="✅", duration="short")
            else:
                st.toast("Training already in progress", icon="❌", duration="short")

    with column_2:
        button_type = "primary" if st.session_state["train_process"] else "secondary"
        if st.button("⏹️ Stop Finetuning", type=button_type, width="stretch", disabled=not bool(st.session_state["train_process"])):
            if st.session_state["train_process"] is not None:
                st.session_state["train_process"].terminate()
                st.session_state["train_process"].wait()
                st.session_state["train_process"] = None
                st.session_state["mode"] = "inference"
                st.toast("Training stopped", icon="✅", duration="short")
                st.rerun()
            else:
                st.toast("No training to stop", icon="❌", duration="short")

    with column_3:
        badge_holder = st.empty()

    # create empty holders
    columns = st.columns(4)
    with columns[0]:
        steps_holder = st.empty()
    with columns[1]:
        format_reward_holder = st.empty()
    with columns[2]:
        correctness_reward_holder = st.empty()
    with columns[3]:
        total_reward_holder = st.empty()
    df_holder = st.empty()

    # parse grpo logs
    if st.session_state["train_process"] is not None:
        while True:
            output = open("/tmp/logs.txt", "r").read().strip()

            logs = []
            for line in output.split("\n"):
                if "{" in line and "}" in line:
                    dict_match = re.search(r"\{[^}]+\}", line)
                    if dict_match:
                        log_dict = eval(dict_match.group())
                        if isinstance(log_dict, dict) and any(k in log_dict for k in [
                            "rewards/format_reward_func/mean",
                            "rewards/correctness_reward_func/mean",
                            "reward",
                            ]):
                            logs.append(log_dict)

            df = pd.DataFrame(logs)
            if "reward" in df.columns:
                steps_holder.metric("Steps", f"{len(df)}" if len(df) > 0 else "N/A")
                format_reward_holder.metric("Format Reward", f"{df['rewards/format_reward_func/mean'].iloc[-1]:.4f}" if len(df) > 0 else "N/A")
                correctness_reward_holder.metric("Correctness Reward", f"{df['rewards/correctness_reward_func/mean'].iloc[-1]:.4f}" if len(df) > 0 else "N/A")
                total_reward_holder.metric("Total Reward", f"{df['reward'].iloc[-1]:.4f}" if len(df) > 0 else "N/A")

                badge_holder.badge("Running", icon=":material/hourglass_arrow_up:", color="green", width="stretch")
            else:
                badge_holder.badge("Loading", icon=":material/hourglass_empty:", color="yellow", width="stretch")

            df_holder.dataframe(df, width="stretch", hide_index=True)
            time.sleep(1)

            if st.session_state["train_process"] is None or st.session_state["train_process"].poll() is not None:
                st.session_state["train_process"].terminate()
                st.session_state["train_process"].wait()
                st.session_state["train_process"] = None
                st.session_state["mode"] = "inference"
                st.toast("Training stopped", icon="✅", duration="short")
                st.rerun()

    else:
        badge_holder.badge("Idle", icon=":material/hourglass_disabled:", color="red", width="stretch")


def inference_section():
    st.header("Image Inference")

    columns = st.columns([3, 3, 1, 2])
    with columns[1]:
        with st.container(border=True, horizontal_alignment="center", vertical_alignment="center"):
            image_holder = st.empty()
            image_holder.image(st.session_state["current_image"])

    with columns[3]:
        if st.button("🎲 Test another sample"):
            while True:
                current_image = random.choice(glob.glob("assets/image_vlm/images/*/*"))
                if current_image != st.session_state["current_image"]:
                    break
            st.session_state["current_image"] = current_image
            image_holder.image(st.session_state["current_image"])

    columns = st.columns(2, gap="small")
    with columns[0]:
        with st.container(border=True):
            st.write("##### :green[Base Qwen2.5-VL-7B]")
            base_generation = st.empty()
            base_generation.write("...")

    with columns[1]:
        with st.container(border=True):
            st.write("##### :green[Finetuned Qwen2.5-VL-7B]")
            finetuned_generation = st.empty()
            finetuned_generation.write("...")

    columns = st.columns([9, 1], gap="small")
    with columns[0]:
        prompt = st.text_input(
            "Prompt Input",
            label_visibility="collapsed",
            key="prompt_input",
            on_change=lambda: st.session_state.update(prompt=st.session_state["prompt_input"])
        )

    with columns[1]:
        if st.button("Generate", width="stretch"):
            if st.session_state.get("prompt", None):
                st.session_state["prompt"] = prompt

                with st.spinner("Running..."):
                    response = start_inference("base")
                base_generation.markdown(response)

                if "finetuned" in st.session_state and "process" in st.session_state["finetuned"]:
                    with st.spinner("Running..."):
                        response = start_inference("finetuned")
                    finetuned_generation.markdown(response)
                else:
                    finetuned_generation.markdown("```No response since there is no finetuned model```")


def start_inference(model_type):
    prompt = st.session_state["prompt"]
    if model_type == "finetuned":
        prompt = (
            f"{prompt}. Also first provide your reasoning or working out"\
            f" on how you would go about identifying the presence of wildfire affected regions between {REASONING_START} and {REASONING_END}"
            f" and then your final answer between {SOLUTION_START} and (put a simple Yes or No here) {SOLUTION_END}"
        )

    response = invoke_vllm_server(
        model_type,
        prompt,
        st.session_state["current_image"],
        st.session_state[model_type]["port"]
    )

    # format response
    if model_type == "finetuned":
        response = response.replace(REASONING_START, "```")
        response = response.replace(REASONING_END, "```")
        
        # Handle solution formatting with proper newline handling
        solution_pattern = f'{re.escape(SOLUTION_START)}(.*?){re.escape(SOLUTION_END)}'
        solution_match = re.search(solution_pattern, response, re.DOTALL)
        if solution_match:
            solution_content = solution_match.group(1).strip()
            response = re.sub(solution_pattern, f"**{solution_content}**", response, flags=re.DOTALL)

    return response


if __name__ == "__main__":
    main()
