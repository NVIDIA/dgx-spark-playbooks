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
docker run -it \
    --rm \
    --gpus all \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/models/vae:/workspace/ComfyUI/models/vae \
    -v $(pwd)/models/loras:/workspace/ComfyUI/models/loras \
    -v $(pwd)/models/checkpoints:/workspace/ComfyUI/models/checkpoints \
    -v $(pwd)/models/text_encoders:/workspace/ComfyUI/models/text_encoders \
    -v $(pwd)/workflows/:/workspace/ComfyUI/user/default/workflows/ \
    flux-comfyui \
    python main.py
