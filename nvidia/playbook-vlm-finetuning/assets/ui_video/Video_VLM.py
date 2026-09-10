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

import json
import yaml
import string
import random

import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torchvision.transforms as T
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoModel
from transformers.trainer_utils import get_last_checkpoint
from torchvision.transforms.functional import InterpolationMode


SCAP_PROMPT = """You are a vision-language assistant analyzing driving videos. You will receive a 5-second video clip of a specific scene. {prompt}

---

### Task 1: Dense Caption
Generate a 2 sentence caption describing:
- Ego vehicle behavior
- Interactions with other vehicles or pedestrians

Focus on **what happens**, **when**, and **who/what is involved**, using only visible information and metadata.

---

### Task 2: Structured JSON
Generate the caption from the perspective of the ego vehicle in a structured JSON object with:

- `"caption"`: from Task 1  
- `"event_type"`: "collision" | "near_miss" | "no_incident"  
- `"rule_violations"`: choose relevant items from ["speeding", "failure_to_yield", "ignoring_traffic_signs"]  
- `"intended_action"`: "turn_left" | "turn_right" | "change_lanes"  
- `"traffic_density"`: "low" | "high"  
- `"visibility"`: "good" | "bad"  
- `"scene"`: "Urban" | "Sub-urban" | "Rural" | "Highway"

**Rules:**
1. Use only visible info and metadata.  
2. Do not invent details.  
3. Include all fields; enum values must match allowed options.  
4. Output a single valid JSON object—no extra text or markdown.  
"""


def random_id():
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=8)).lower()


def initialize_session_state(base_model, finetuned_model):
    # Initialize page-specific session state
    st.session_state["base_video_vlm"] = st.session_state.get("base_video_vlm", base_model)
    st.session_state["finetuned_video_vlm"] = st.session_state.get("finetuned_video_vlm", finetuned_model)
    st.session_state["current_sample"] = st.session_state.get("current_sample", None)
    st.session_state["df"] = st.session_state.get("df",
        pd.DataFrame(columns=[
            "Driver ID",
            "Event Type",
            "Rule Violations",
            "Intended Action",
            "Traffic Density",
            "Driving Scene",
            "Visibility"]))


def load_config():
    config_key = "config_video_vlm"
    if getattr(st.session_state, config_key, None) is None:
        with open("src/video_vlm_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        setattr(st.session_state, config_key, config)
    else:
        config = getattr(st.session_state, config_key)

    return config


@st.cache_resource
def initialize_model(model_path):
    model = InternVLModel(model_path)
    return {"model": model}


def main():
    # set page ui
    st.title(":green[Video VLM on DGX Spark]")
    st.caption("Driver Behavior Analysis via Structured Video Captioning")
    # st.page_link("https://github.com/your-username/your-repo", label="GitHub", icon=":material/github:")

    # load css
    with open("src/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # load resources
    config = load_config()
    if st.session_state.get("base_video_vlm", None) is None:
        st.toast("Loading model", icon="⏳", duration="short")
    base_model = initialize_model(config["inference"]["model_id"])
    finetuned_model_path = get_last_checkpoint(config["inference"]["finetuned_model_id"])
    if finetuned_model_path is not None:
        finetuned_model = initialize_model(finetuned_model_path)
    else:
        finetuned_model = {"model": None}
    if st.session_state.get("base_video_vlm", None) is None:
        st.toast("Model loaded", icon="✅", duration="short")
    initialize_session_state(base_model, finetuned_model)

    # gallery section
    st.markdown("---")
    st.header("Dashcam Gallery")

    columns = st.columns([4, 1, 1, 4, 1, 1, 4, 1], gap="small")

    with columns[0]:
        st.image("assets/video_vlm/thumbnails/1.png")

    with columns[1]:
        if st.button(":material/file_open:", key="video_1"):
            st.session_state["current_sample"] = "assets/video_vlm/videos/1.mp4"

    with columns[3]:
        st.image("assets/video_vlm/thumbnails/2.png")

    with columns[4]:
        if st.button(":material/file_open:", key="video_2"):
            st.session_state["current_sample"] = "assets/video_vlm/videos/2.mp4"

    with columns[6]:
        st.image("assets/video_vlm/thumbnails/3.png")

    with columns[7]:
        if st.button(":material/file_open:", key="video_3"):
            st.session_state["current_sample"] = "assets/video_vlm/videos/3.mp4"

    # inference section
    st.markdown("---")
    st.header("Video Inference")

    with st.container(border=True):
        if st.session_state["current_sample"]:
            st.video(st.session_state["current_sample"], autoplay=True, loop=True, muted=True)
        else:
            st.write(":gray[Please select a video from the dashcam gallery.]")

    columns = st.columns(2, gap="small")
    with columns[0]:
        with st.container(border=True):
            st.write("##### :green[Base InternVL3-8B]")
            base_generation = st.empty()
            base_generation.write("...")

    with columns[1]:
        with st.container(border=True):
            st.write("##### :green[Finetuned InternVL3-8B]")
            finetuned_generation = st.empty()
            finetuned_generation.write("...")

    columns = st.columns([9, 1], gap="small")
    with columns[0]:
        prompt = st.text_input(
            "Prompt Input",
            label_visibility="collapsed",
            key="prompt_input",
            on_change=lambda: st.session_state.update(prompt=st.session_state.prompt_input)
        )

    with columns[1]:
        if st.button("Generate", width="stretch"):
            if st.session_state.get("prompt", None):
                st.session_state["prompt"] = prompt

                with st.spinner("Running..."):
                    response = start_inference("base")
                base_generation.markdown(response)

                if st.session_state["finetuned_video_vlm"].get("model", None) is not None:
                    with st.spinner("Running..."):
                        response = start_inference("finetuned")
                    finetuned_generation.markdown(response)

                    response = json.loads(response[7: -3].strip())
                    response["caption"] = random_id() # replace caption with driver id
                    st.session_state["df"].loc[len(st.session_state["df"])] = list(response.values())
                else:
                    finetuned_generation.markdown("```No response since there is no finetuned model```")

    # data analysis section
    st.markdown("---")
    st.header("Data Analysis")

    with st.container(border=True):
        st.dataframe(st.session_state["df"])


class InternVLVideoProcessor:
    def __init__(self):
        self.frame_size = 448

        self.transform = T.Compose([
            T.Resize(self.frame_size, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.frame_size * self.frame_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, frame, min_num=1, max_num=12, use_thumbnail=False):
        orig_width, orig_height = frame.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, self.frame_size)

        # calculate the target width and height
        target_width = self.frame_size * target_aspect_ratio[0]
        target_height = self.frame_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = frame.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.frame_size)) * self.frame_size,
                (i // (target_width // self.frame_size)) * self.frame_size,
                ((i % (target_width // self.frame_size)) + 1) * self.frame_size,
                ((i // (target_width // self.frame_size)) + 1) * self.frame_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = frame.resize((self.frame_size, self.frame_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_video(self, video_path, num_frames, start_frame=None, end_frame=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(vr) - 1

        # sample a random number of equally-spaced frames from the video
        frame_indices = np.linspace(
            start_frame,
            end_frame,
            num_frames,
            dtype=int
        )

        pixel_values_list, num_patches_list = [], []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, use_thumbnail=True, max_num=1)
            pixel_values = [self.transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list


class InternVLModel:
    def __init__(self, model_path):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        self.processor = InternVLVideoProcessor()
        self.generation_config = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            top_k=1,
            top_p=1.0)

    @torch.no_grad()
    def infer(self, video_path, prompt, sampling_mode, num_frames=32, chunk_duration=2):
        if sampling_mode == "default":
            return self._infer_default(video_path, prompt, num_frames)
        elif sampling_mode == "real-time":
            return self._infer_realtime(video_path, prompt, num_frames, chunk_duration)
    
    def _infer_default(self, video_path, prompt, num_frames):
        pixel_values, num_patches_list = self.processor.load_video(video_path, num_frames)
        pixel_values = pixel_values.to(device=self.model.device, dtype=torch.bfloat16)

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        prompt = f"{video_prefix}{prompt}"

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            self.generation_config,
            num_patches_list=num_patches_list
        )

        return response

    def _infer_realtime(self, video_path, prompt, num_frames, chunk_duration):
        video = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = video.get_avg_fps()
        total_frames = len(video)
        frames_per_chunk = int(chunk_duration * fps)

        for start_frame in range(0, total_frames, frames_per_chunk):
            end_frame = start_frame + frames_per_chunk
            if end_frame > total_frames:
                return
            
            pixel_values, num_patches_list = self.processor.load_video(video_path, num_frames, start_frame, end_frame)
            pixel_values = pixel_values.to(device=self.model.device, dtype=torch.bfloat16)

            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            prompt = f"{video_prefix}{prompt}"

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                self.generation_config,
                num_patches_list=num_patches_list
            )

            yield response


@torch.no_grad()
def start_inference(model_type):
    # define prompt
    prompt = st.session_state["prompt"]
    if model_type == "finetuned":
        prompt = SCAP_PROMPT.format(prompt=prompt)

    print(print(model_type))
    response = st.session_state[f"{model_type}_video_vlm"]["model"].infer(
        st.session_state["current_sample"],
        prompt,
        num_frames=st.session_state["config_video_vlm"]["inference"]["num_frames"],
        sampling_mode=st.session_state["config_video_vlm"]["inference"]["sampling_mode"]
    )

    if model_type == "finetuned":
        response = f"```json\n{json.dumps(json.loads(response), indent=2)}\n```"

    return response


if __name__ == "__main__":
    main()
