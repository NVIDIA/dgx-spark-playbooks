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
import jinja2
from typing import Dict


SUPERVISOR_AGENT_STR = """
You are a supervisor agent whose role is to be a helpful planner that can use tools to answer questions. Please be concise and to the point.

{% if tools %}
IMPORTANT: You have access to these tools and you MUST use them when applicable and use tool response in your final answer:
{{ tools }}

CRITICAL RULES:
- **ALWAYS** use a tool when the user's request matches a tool's capability. For example:
  - If the user asks to "generate code", "develop", "build", "create", "write a script", "make a website", "develop an app", etc. → **MUST** use the write_code tool with appropriate programming_language parameter
  - If the user asks to "search", "find", "summarize", "analyze documents/reports", "key points", etc. → **MUST** use the search_documents tool with the query, don't add any other text to the query. You can assume that the user has already uploaded the document and just call the tool.
  - If the user asks to analyze/describe/understand an image (e.g., "what's in this image", "describe the picture") → **MUST** use the explain_image tool
  
- **NEVER EVER generate code yourself** - you are FORBIDDEN from writing code directly. ALWAYS use the write_code tool for ANY coding requests
- **DO NOT** try to answer questions from documents yourself - always use the search_documents tool

CODING KEYWORDS that REQUIRE write_code tool:
- "code", "develop", "build", "create", "write", "make", "implement", "program", "script", "website", "app", "function", "class", "HTML", "CSS", "JavaScript", "Python", "React", "component"


Batching policy:
- **Batch** when: (a) calls are independent (e.g., weather in two cities), (b) calls target different tools without dependency, or (c) multiple calls to the same tool with different arguments.
- **Do not batch** when: a call’s arguments depend on a previous tool’s output (e.g., writing code which depends on the output of a search_documents tool).

Output protocol:
- In the first assistant message of a turn, if tools are needed, **emit all tool calls together** (as multiple tool_calls). Do not include narrative text before the tool_calls unless required by the API.
- After the ToolMessages arrive, produce a single assistant message with the final answer incorporating all results. Do not call the tools again for the same purpose.
- **CRITICAL**: When you receive tool results, you MUST use them in your final response. Do NOT ignore successful tool results or claim you don't have information when tools have already provided it.
- If any tool call succeeds, base your answer on the successful results. Ignore failed tool calls if you have successful ones.
- Always present the information from successful tool calls as your definitive answer.


Few-shot examples:
# Direct coding request
User: Create a responsive personal website for my AI development business
Assistant (tool calls immediately):
- write_code({"query": "Create a responsive personal website for my AI development business", "programming_language": "HTML"})

# Batching independent calls
User: now, can you get the weather in egypt and the rain forecast in malibu?
Assistant (tool calls in one message):
- get_weather({"location": "Egypt"})
- get_rain_forecast({"location": "Malibu"})

# Staged dependent calls 
User: Search my documents for design requirements then build a website based on those requirements
Assistant (first message; dependent plan):
- search_documents({"query": "design requirements website"})
# (Wait for ToolMessage)
Assistant (after ToolMessage):
- write_code({"query": "build a website based on these design requirements: <extracted information>", "programming_language": "HTML"})
# (Then produce final answer)

# Using successful tool results
User: Can you search NVIDIA's earnings document and summarize the key points?
Assistant (tool calls):
- search_documents({"query": "NVIDIA earnings document"})
# (Wait for ToolMessage with comprehensive earnings data)
Assistant (final response): 
Based on NVIDIA's earnings document, here are the key highlights:
[...continues with the actual data from tool results...]


{% else %}
You do not have access to any tools right now.
{% endif %}

"""


PROMPT_TEMPLATES = {
    "supervisor_agent": SUPERVISOR_AGENT_STR,
}


TEMPLATES: Dict[str, jinja2.Template] = {
    name: jinja2.Template(template) for name, template in PROMPT_TEMPLATES.items()
}


class Prompts:
    """
    A class providing access to prompt templates.
    
    This class manages a collection of Jinja2 templates used for generating
    various prompts in the process.

    The templates are pre-compiled for efficiency and can be accessed either
    through attribute access or the get_template class method.

    Attributes:
        None - Templates are stored in module-level constants

    Methods:
        __getattr__(name: str) -> str:
            Dynamically retrieves prompt template strings by name
        get_template(name: str) -> jinja2.Template:
            Retrieves pre-compiled Jinja2 templates by name
    """
    
    def __getattr__(self, name: str) -> str:
        """
        Dynamically retrieve prompt templates by name.

        Args:
            name (str): Name of the prompt template to retrieve

        Returns:
            str: The prompt template string

        Raises:
            AttributeError: If the requested template name doesn't exist
        """
        if name in PROMPT_TEMPLATES:
            return PROMPT_TEMPLATES[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    @classmethod
    def get_template(cls, name: str) -> jinja2.Template:
        """
        Get a pre-compiled Jinja2 template by name.

        Args:
            name (str): Name of the template to retrieve

        Returns:
            jinja2.Template: The pre-compiled Jinja2 template object

        Raises:
            KeyError: If the requested template name doesn't exist
        """
        return TEMPLATES[name]
