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
from pydantic import BaseModel
from typing import Optional, List

class ChatConfig(BaseModel):
    sources: List[str]
    models :  List[str]
    selected_model: Optional[str] = None
    selected_sources: Optional[List[str]] = None
    current_chat_id: Optional[str] = None

class ChatIdRequest(BaseModel):
    chat_id: str

class ChatRenameRequest(BaseModel):
    chat_id: str
    new_name: str

class SelectedModelRequest(BaseModel):
    model: str      
