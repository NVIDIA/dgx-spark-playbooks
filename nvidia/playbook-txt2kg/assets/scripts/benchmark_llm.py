#!/usr/bin/env python3
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
LLM Benchmark Script: vLLM vs Ollama Performance Comparison
Compares performance metrics between vLLM and Ollama deployments
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
import sys

@dataclass
class BenchmarkResult:
    service: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    tokens_per_second: float
    first_token_time: float = 0.0
    error: str = ""

class LLMBenchmark:
    def __init__(self):
        self.vllm_url = "http://localhost:8001"
        self.ollama_url = "http://localhost:11434"
        
    async def test_vllm(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int = 100) -> BenchmarkResult:
        """Test vLLM performance"""
        start_time = time.time()
        
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            async with session.post(f"{self.vllm_url}/v1/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return BenchmarkResult(
                        service="vLLM",
                        model="Llama-3.1-8B-Instruct",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        response_time=0,
                        tokens_per_second=0,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                
                result = await response.json()
                end_time = time.time()
                
                response_time = end_time - start_time
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
                
                return BenchmarkResult(
                    service="vLLM",
                    model="Llama-3.1-8B-Instruct",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    response_time=response_time,
                    tokens_per_second=tokens_per_second
                )
                
        except Exception as e:
            return BenchmarkResult(
                service="vLLM",
                model="Llama-3.1-8B-Instruct",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    async def test_ollama(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int = 100) -> BenchmarkResult:
        """Test Ollama performance"""
        start_time = time.time()
        
        payload = {
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            async with session.post(f"{self.ollama_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return BenchmarkResult(
                        service="Ollama",
                        model="llama3.1:8b",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        response_time=0,
                        tokens_per_second=0,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                
                result = await response.json()
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Ollama response format
                prompt_eval_count = result.get("prompt_eval_count", 0)
                eval_count = result.get("eval_count", 0)
                total_tokens = prompt_eval_count + eval_count
                
                tokens_per_second = eval_count / response_time if response_time > 0 else 0
                
                return BenchmarkResult(
                    service="Ollama",
                    model="llama3.1:8b",
                    prompt_tokens=prompt_eval_count,
                    completion_tokens=eval_count,
                    total_tokens=total_tokens,
                    response_time=response_time,
                    tokens_per_second=tokens_per_second
                )
                
        except Exception as e:
            return BenchmarkResult(
                service="Ollama",
                model="llama3.1:8b",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    async def run_single_test(self, prompt: str, max_tokens: int = 100) -> tuple[BenchmarkResult, BenchmarkResult]:
        """Run a single test comparing both services"""
        async with aiohttp.ClientSession() as session:
            # Test both services concurrently
            vllm_task = self.test_vllm(session, prompt, max_tokens)
            ollama_task = self.test_ollama(session, prompt, max_tokens)
            
            vllm_result, ollama_result = await asyncio.gather(vllm_task, ollama_task)
            return vllm_result, ollama_result
    
    async def run_benchmark(self, prompts: List[str], max_tokens: int = 100, runs_per_prompt: int = 3) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark"""
        results = {"vLLM": [], "Ollama": []}
        
        print(f"Running benchmark with {len(prompts)} prompts, {runs_per_prompt} runs each...")
        print(f"Max tokens per completion: {max_tokens}")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}/{len(prompts)}: {prompt[:50]}...")
            
            for run in range(runs_per_prompt):
                print(f"  Run {run + 1}/{runs_per_prompt}...", end=" ")
                
                vllm_result, ollama_result = await self.run_single_test(prompt, max_tokens)
                
                results["vLLM"].append(vllm_result)
                results["Ollama"].append(ollama_result)
                
                # Print quick results
                if vllm_result.error:
                    print(f"vLLM: ERROR - {vllm_result.error}")
                else:
                    print(f"vLLM: {vllm_result.response_time:.2f}s ({vllm_result.tokens_per_second:.1f} tok/s)", end=" | ")
                
                if ollama_result.error:
                    print(f"Ollama: ERROR - {ollama_result.error}")
                else:
                    print(f"Ollama: {ollama_result.response_time:.2f}s ({ollama_result.tokens_per_second:.1f} tok/s)")
                
                # Small delay between runs
                await asyncio.sleep(1)
        
        return results
    
    def analyze_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Analyze and print benchmark results"""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS ANALYSIS")
        print("=" * 80)
        
        for service_name, service_results in results.items():
            print(f"\n{service_name} Results:")
            print("-" * 40)
            
            # Filter out errors
            valid_results = [r for r in service_results if not r.error]
            error_results = [r for r in service_results if r.error]
            
            if error_results:
                print(f"Errors: {len(error_results)}/{len(service_results)}")
                for error in set(r.error for r in error_results):
                    print(f"  - {error}")
                print()
            
            if not valid_results:
                print("No valid results to analyze.")
                continue
            
            # Calculate statistics
            response_times = [r.response_time for r in valid_results]
            tokens_per_second = [r.tokens_per_second for r in valid_results]
            completion_tokens = [r.completion_tokens for r in valid_results]
            
            print(f"Valid runs: {len(valid_results)}")
            print(f"Response time (avg): {statistics.mean(response_times):.3f}s")
            print(f"Response time (median): {statistics.median(response_times):.3f}s")
            print(f"Response time (min/max): {min(response_times):.3f}s / {max(response_times):.3f}s")
            print(f"Tokens/second (avg): {statistics.mean(tokens_per_second):.1f}")
            print(f"Tokens/second (median): {statistics.median(tokens_per_second):.1f}")
            print(f"Tokens/second (min/max): {min(tokens_per_second):.1f} / {max(tokens_per_second):.1f}")
            print(f"Completion tokens (avg): {statistics.mean(completion_tokens):.1f}")
        
        # Comparison
        vllm_valid = [r for r in results["vLLM"] if not r.error]
        ollama_valid = [r for r in results["Ollama"] if not r.error]
        
        if vllm_valid and ollama_valid:
            print("\n" + "=" * 40)
            print("PERFORMANCE COMPARISON")
            print("=" * 40)
            
            vllm_avg_response = statistics.mean([r.response_time for r in vllm_valid])
            ollama_avg_response = statistics.mean([r.response_time for r in ollama_valid])
            
            vllm_avg_tokens_sec = statistics.mean([r.tokens_per_second for r in vllm_valid])
            ollama_avg_tokens_sec = statistics.mean([r.tokens_per_second for r in ollama_valid])
            
            if vllm_avg_response < ollama_avg_response:
                speedup = ollama_avg_response / vllm_avg_response
                print(f"🏆 vLLM is {speedup:.2f}x FASTER in response time")
            else:
                speedup = vllm_avg_response / ollama_avg_response
                print(f"🏆 Ollama is {speedup:.2f}x FASTER in response time")
            
            if vllm_avg_tokens_sec > ollama_avg_tokens_sec:
                throughput_ratio = vllm_avg_tokens_sec / ollama_avg_tokens_sec
                print(f"🚀 vLLM has {throughput_ratio:.2f}x HIGHER throughput")
            else:
                throughput_ratio = ollama_avg_tokens_sec / vllm_avg_tokens_sec
                print(f"🚀 Ollama has {throughput_ratio:.2f}x HIGHER throughput")

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs Ollama")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer prompts")
    
    args = parser.parse_args()
    
    # Test prompts
    if args.quick:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
        ]
    else:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot learning to paint.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?",
        ]
    
    benchmark = LLMBenchmark()
    
    try:
        results = asyncio.run(benchmark.run_benchmark(prompts, args.max_tokens, args.runs))
        benchmark.analyze_results(results)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
