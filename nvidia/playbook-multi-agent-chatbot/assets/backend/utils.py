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
"""Utility functions for file processing and message conversion."""

import json
import os
import time
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

from logger import logger
from vector_store import VectorStore


async def process_and_ingest_files_background(
    file_info: List[dict], 
    vector_store: VectorStore, 
    config_manager, 
    task_id: str, 
    indexing_tasks: Dict[str, str]
) -> None:
    """Process and ingest files in the background.
    
    Args:
        file_info: List of file dictionaries with 'filename' and 'content' keys
        vector_store: VectorStore instance for document indexing
        config_manager: ConfigManager instance for updating sources
        task_id: Unique identifier for this processing task
        indexing_tasks: Dictionary to track task status
    """
    try:
        logger.debug({
            "message": "Starting background file processing",
            "task_id": task_id,
            "file_count": len(file_info)
        })
        
        indexing_tasks[task_id] = "saving_files"
        
        permanent_dir = os.path.join("uploads", task_id)
        os.makedirs(permanent_dir, exist_ok=True)
        
        file_paths = []
        file_names = []
        
        for info in file_info:
            try:
                file_name = info["filename"]
                content = info["content"]
                
                file_path = os.path.join(permanent_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(content)
                
                file_paths.append(file_path)
                file_names.append(file_name)
                
                logger.debug({
                    "message": "Saved file",
                    "task_id": task_id,
                    "filename": file_name,
                    "path": file_path
                })
            except Exception as e:
                logger.error({
                    "message": f"Error saving file {info['filename']}",
                    "task_id": task_id,
                    "filename": info['filename'],
                    "error": str(e)
                }, exc_info=True)
        
        indexing_tasks[task_id] = "loading_documents"
        logger.debug({"message": "Loading documents", "task_id": task_id})
        
        try:
            documents = vector_store._load_documents(file_paths)
            
            logger.debug({
                "message": "Documents loaded, starting indexing",
                "task_id": task_id,
                "document_count": len(documents)
            })
            
            indexing_tasks[task_id] = "indexing_documents"
            vector_store.index_documents(documents)
            
            if file_names:
                config = config_manager.read_config()
                
                config_updated = False
                for file_name in file_names:
                    if file_name not in config.sources:
                        config.sources.append(file_name)
                        config_updated = True
                
                if config_updated:
                    config_manager.write_config(config)
                    logger.debug({
                        "message": "Updated config with new sources",
                        "task_id": task_id,
                        "sources": config.sources
                    })
            
            indexing_tasks[task_id] = "completed"
            logger.debug({
                "message": "Background processing and indexing completed successfully",
                "task_id": task_id
            })
        except Exception as e:
            indexing_tasks[task_id] = f"failed_during_indexing: {str(e)}"
            logger.error({
                "message": "Error during document loading or indexing",
                "task_id": task_id,
                "error": str(e)
            }, exc_info=True)
            
    except Exception as e:
        indexing_tasks[task_id] = f"failed: {str(e)}"
        logger.error({
            "message": "Error in background processing",
            "task_id": task_id,
            "error": str(e)
        }, exc_info=True)


def convert_langgraph_messages_to_openai(messages: List) -> List[Dict[str, Any]]:
    """Convert LangGraph message objects to OpenAI API format.
    
    Args:
        messages: List of LangGraph message objects
        
    Returns:
        List of dictionaries in OpenAI API format
    """
    openai_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            openai_messages.append({
                "role": "user", 
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            openai_msg = {
                "role": "assistant", 
                "content": msg.content or ""
            }
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                openai_msg["tool_calls"] = []
                for tc in msg.tool_calls:
                    openai_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    })
            openai_messages.append(openai_msg)
        elif isinstance(msg, ToolMessage):
            openai_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
    
    return openai_messages
