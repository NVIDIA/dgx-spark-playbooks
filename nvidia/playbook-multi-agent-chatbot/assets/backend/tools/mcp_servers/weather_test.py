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

"""
Weather Test MCP Server

A lightweight Model Context Protocol (MCP) server designed for testing and demonstration purposes.
This module provides mock weather tools that return humorous, fake responses rather than real 
weather data.

Features:
    - Mock weather data retreival for any location
    - Example implementation of MCP tool registration with local functions
    - Demonstration of FastMCP server setup

Tools provided:
    - get_weather(location): Returns mock weather information
    - get_rain_forecast(location): Returns mock rain forecast data

Usage:
  - Usage in Chatbot Spark:
    The server is ran on project startup by MCPClient. SEe client.py for more details
  - Standalone Usage:
    Run as standalone script to start the MCP server:
    $ python weather_test.py
"""
import time
import os

from langchain_core.tools import tool, Tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("weather-tools")


@mcp.tool()
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    else:
        return f"The weather is spooky with a chance of gremlins in {location}"


@mcp.tool()
def get_rain_forecast(location: str):
    """Call to get the rain forecast from a specific location."""
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's going to rain cats and dogs in San Francisco tomorrow."
    else:
        return f"It is raining muffins in {location}"


if __name__ == "__main__":
    print(f'running {mcp.name} MCP server')
    mcp.run(transport="stdio")