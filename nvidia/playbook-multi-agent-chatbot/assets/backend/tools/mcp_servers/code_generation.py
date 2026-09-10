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

import asyncio
from typing import Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

mcp = FastMCP("Code Generation")
model_name = "deepseek-coder:6.7b"


@mcp.tool()
async def write_code(query: str, programming_language: str):
    """This tool is used to write complete code.
    
    Args:
        query: The natural language description of the code to be generated.
        programming_language: The programming language for the code generation (e.g., 'Python', 'JavaScript', 'HTML', 'CSS', 'Go').
        
    Returns:
        The generated code.
    """
    model_client = AsyncOpenAI(
        base_url="http://deepseek-coder:8000/v1",
        api_key="ollama"
    )
    
    system_prompt = f"""You are an expert coder specializing in {programming_language}.
    Given a user request, generate clean, efficient {programming_language} code that accomplishes the specified task.
    Always provide the full code generation so the user can copy and paste a fully working example.
    Return just the raw code, with no markdown formatting, explanations, or any other text.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    response = await model_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
    )
    
    generated_code = response.choices[0].message.content
    return generated_code.strip()


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")