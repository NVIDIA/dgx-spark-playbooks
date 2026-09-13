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
Knowledge Graph Extraction Benchmark: vLLM vs Ollama
Realistic benchmark based on the txt2kg codebase use case
Tests triple extraction from 512-character text chunks
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
import subprocess
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class KGBenchmarkResult:
    service: str
    model: str
    text_chunk: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    tokens_per_second: float
    extracted_triples: List[Dict] = None
    raw_response: str = ""
    error: str = ""

class KGExtractionBenchmark:
    def __init__(self):
        self.vllm_url = "http://localhost:8001"
        self.ollama_url = "http://localhost:11434"
        self.vllm_dir = "/home/nvidia/txt2kg/txt2kg/deploy/services/vllm"
        self.ollama_dir = "/home/nvidia/txt2kg/txt2kg/deploy/services/ollama"
        
        # Real prompts from the txt2kg codebase
        self.system_prompt = """You are a knowledge graph builder that extracts structured information from text.
Extract subject-predicate-object triples from the following text.

Guidelines:
- Extract only factual triples present in the text
- Normalize entity names to their canonical form
- Return results in JSON format as an array of objects with "subject", "predicate", "object" fields
- Each triple should represent a clear relationship between two entities
- Focus on the most important relationships in the text"""

        # Alternative system prompt from the codebase
        self.alternative_system_prompt = """You are an expert that can extract knowledge triples with the form `('entity', 'relation', 'entity)` from a text, mainly using entities from the entity list given by the user. Keep relations 2 words max.
Separate each with a new line. Do not output anything else (no notes, no explanations, etc)."""
        
    def get_realistic_text_chunks(self) -> List[str]:
        """Generate realistic 512-character text chunks for knowledge extraction"""
        chunks = [
            # Scientific/Technical text
            """Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California, and is known for developing innovative consumer electronics like the iPhone, iPad, and Mac computers. Tim Cook became the CEO after Steve Jobs passed away in 2011. Apple's market capitalization exceeded $3 trillion in 2022, making it one of the most valuable companies in the world. The company operates retail stores globally and has a strong focus on design and user experience.""",
            
            # Business/Corporate text
            """Tesla Motors was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined the company as chairman of the board in 2004 and became CEO in 2008. Tesla is headquartered in Austin, Texas, and manufactures electric vehicles, energy storage systems, and solar panels. The company's Gigafactory in Nevada produces lithium-ion batteries for its vehicles. Tesla went public in 2010 and has become a leader in the electric vehicle market with models like the Model S, Model 3, and Model Y.""",
            
            # Academic/Research text
            """The University of California, Berkeley was established in 1868 and is located in Berkeley, California. It is part of the University of California system and is known for its research programs in computer science, engineering, and physics. Notable alumni include Steve Wozniak, co-founder of Apple, and Eric Schmidt, former CEO of Google. The university operates the Lawrence Berkeley National Laboratory and has produced numerous Nobel Prize winners. Berkeley's computer science department developed the BSD Unix operating system.""",
            
            # Historical/Biographical text
            """Albert Einstein was born in Ulm, Germany in 1879 and later moved to Princeton, New Jersey. He developed the theory of relativity, which revolutionized physics and our understanding of space and time. Einstein worked at Princeton University's Institute for Advanced Study from 1933 until his death in 1955. He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Einstein's famous equation E=mc² demonstrates the relationship between mass and energy.""",
            
            # Medical/Healthcare text
            """The World Health Organization (WHO) is a specialized agency of the United Nations responsible for international public health. It was established in 1948 and is headquartered in Geneva, Switzerland. Dr. Tedros Adhanom Ghebreyesus serves as the current Director-General. WHO coordinates international health work, monitors disease outbreaks, and provides technical assistance to countries. During the COVID-19 pandemic, WHO played a crucial role in coordinating the global response and providing guidance on vaccines and treatments.""",
            
            # Technology/Innovation text
            """Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford University. The company is now part of Alphabet Inc. and is headquartered in Mountain View, California. Google's search engine processes billions of queries daily and the company has expanded into cloud computing, artificial intelligence, and autonomous vehicles. Sundar Pichai became CEO of Google in 2015. The company's other products include Gmail, YouTube, Android, and Google Cloud Platform."""
        ]
        
        # Ensure each chunk is approximately 512 characters
        processed_chunks = []
        for chunk in chunks:
            if len(chunk) > 512:
                # Truncate to 512 characters at word boundary
                truncated = chunk[:512]
                last_space = truncated.rfind(' ')
                if last_space > 400:  # Ensure we don't cut too much
                    chunk = truncated[:last_space] + "."
                else:
                    chunk = truncated
            processed_chunks.append(chunk)
            
        return processed_chunks
    
    def run_command(self, cmd: str, cwd: str = None) -> tuple[int, str]:
        """Run a shell command and return exit code and output"""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=120
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "Command timed out"
        except Exception as e:
            return -1, str(e)
    
    def stop_all_services(self):
        """Stop both vLLM and Ollama services"""
        print("🛑 Stopping all services...")
        
        # Stop vLLM
        exit_code, output = self.run_command("docker compose down", self.vllm_dir)
        if exit_code != 0:
            print(f"Warning: Failed to stop vLLM: {output}")
        
        # Stop Ollama
        exit_code, output = self.run_command("docker compose down", self.ollama_dir)
        if exit_code != 0:
            print(f"Warning: Failed to stop Ollama: {output}")
        
        # Wait for services to fully stop
        time.sleep(10)
        print("✅ All services stopped")
    
    def start_vllm(self) -> bool:
        """Start vLLM service and wait for it to be ready"""
        print("🚀 Starting vLLM service...")
        
        # Start the service
        exit_code, output = self.run_command("bash -c 'source .env && docker compose up -d'", self.vllm_dir)
        if exit_code != 0:
            print(f"❌ Failed to start vLLM: {output}")
            return False
        
        # Wait for service to be ready (extended timeout for 70B model)
        print("⏳ Waiting for vLLM to be ready (70B model may take 10-15 minutes)...")
        for i in range(180):  # Wait up to 15 minutes for 70B model
            try:
                response = subprocess.run(
                    ["curl", "-s", f"{self.vllm_url}/health"],
                    capture_output=True,
                    timeout=5
                )
                if response.returncode == 0:
                    print("✅ vLLM is ready!")
                    return True
            except:
                pass
            
            time.sleep(5)
            if i % 6 == 0:  # Print progress every 30 seconds
                print(f"   Still waiting... ({i*5}s)")
        
        print("❌ vLLM failed to start within timeout")
        return False
    
    def start_ollama(self) -> bool:
        """Start Ollama service and wait for it to be ready"""
        print("🚀 Starting Ollama service...")
        
        # Start the service
        exit_code, output = self.run_command("docker compose up -d", self.ollama_dir)
        if exit_code != 0:
            print(f"❌ Failed to start Ollama: {output}")
            return False
        
        # Wait for service to be ready
        print("⏳ Waiting for Ollama to be ready...")
        for i in range(24):  # Wait up to 2 minutes
            try:
                response = subprocess.run(
                    ["curl", "-s", f"{self.ollama_url}/api/tags"],
                    capture_output=True,
                    timeout=5
                )
                if response.returncode == 0:
                    print("✅ Ollama is ready!")
                    return True
            except:
                pass
            
            time.sleep(5)
            if i % 6 == 0:  # Print progress every 30 seconds
                print(f"   Still waiting... ({i*5}s)")
        
        print("❌ Ollama failed to start within timeout")
        return False
    
    def extract_triples_from_response(self, response_text: str) -> List[Dict]:
        """Extract triples from LLM response"""
        triples = []
        try:
            # Try to parse as JSON first
            json_match = None
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_match = response_text[start:end]
            
            if json_match:
                triples = json.loads(json_match)
            else:
                # Fallback: parse line by line for tuple format
                lines = response_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if '(' in line and ')' in line and ',' in line:
                        # Extract tuple format ('entity', 'relation', 'entity')
                        try:
                            # Simple tuple parsing
                            content = line[line.find('(')+1:line.rfind(')')]
                            parts = [part.strip().strip("'\"") for part in content.split(',')]
                            if len(parts) >= 3:
                                triples.append({
                                    'subject': parts[0],
                                    'predicate': parts[1],
                                    'object': parts[2]
                                })
                        except:
                            continue
        except Exception as e:
            print(f"Warning: Failed to parse triples: {e}")
        
        return triples
    
    async def test_vllm_kg_extraction(self, session: aiohttp.ClientSession, text_chunk: str) -> KGBenchmarkResult:
        """Test vLLM knowledge graph extraction"""
        start_time = time.time()
        
        # Use chat completions format for better results
        payload = {
            "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": f"Extract triples from this text:\n\n{text_chunk}"
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            async with session.post(f"{self.vllm_url}/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return KGBenchmarkResult(
                        service="vLLM",
                        model="Llama-3.1-8B-Instruct-FP8",
                        text_chunk=text_chunk[:100] + "...",
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
                
                # Extract response text
                raw_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                extracted_triples = self.extract_triples_from_response(raw_response)
                
                return KGBenchmarkResult(
                    service="vLLM",
                    model="Llama-3.3-70B-Instruct-FP4",
                    text_chunk=text_chunk[:100] + "...",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    response_time=response_time,
                    tokens_per_second=tokens_per_second,
                    extracted_triples=extracted_triples,
                    raw_response=raw_response
                )
                
        except Exception as e:
            return KGBenchmarkResult(
                service="vLLM",
                model="Llama-3.3-70B-Instruct-FP4",
                text_chunk=text_chunk[:100] + "...",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    async def test_ollama_kg_extraction(self, session: aiohttp.ClientSession, text_chunk: str) -> KGBenchmarkResult:
        """Test Ollama knowledge graph extraction"""
        start_time = time.time()
        
        payload = {
            "model": "llama3.1:8b",
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Extract triples from this text:\n\n{text_chunk}"
                }
            ],
            "stream": False,
            "options": {
                "num_predict": 1024,
                "temperature": 0.1
            }
        }
        
        try:
            async with session.post(f"{self.ollama_url}/api/chat", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return KGBenchmarkResult(
                        service="Ollama",
                        model="llama3.1:8b",
                        text_chunk=text_chunk[:100] + "...",
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
                
                # Extract response text
                raw_response = result.get("message", {}).get("content", "")
                extracted_triples = self.extract_triples_from_response(raw_response)
                
                return KGBenchmarkResult(
                    service="Ollama",
                    model="llama3.1:8b",
                    text_chunk=text_chunk[:100] + "...",
                    prompt_tokens=prompt_eval_count,
                    completion_tokens=eval_count,
                    total_tokens=total_tokens,
                    response_time=response_time,
                    tokens_per_second=tokens_per_second,
                    extracted_triples=extracted_triples,
                    raw_response=raw_response
                )
                
        except Exception as e:
            return KGBenchmarkResult(
                service="Ollama",
                model="llama3.1:8b",
                text_chunk=text_chunk[:100] + "...",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    async def benchmark_service(self, service: str, text_chunks: List[str], runs_per_chunk: int = 2) -> List[KGBenchmarkResult]:
        """Benchmark a single service for knowledge graph extraction"""
        results = []
        
        print(f"\n{'='*70}")
        print(f"BENCHMARKING {service.upper()} - KNOWLEDGE GRAPH EXTRACTION")
        print(f"{'='*70}")
        
        async with aiohttp.ClientSession() as session:
            for i, chunk in enumerate(text_chunks, 1):
                print(f"\nText Chunk {i}/{len(text_chunks)} ({len(chunk)} chars): {chunk[:80]}...")
                
                for run in range(runs_per_chunk):
                    print(f"  Run {run + 1}/{runs_per_chunk}...", end=" ")
                    
                    if service == "vLLM":
                        result = await self.test_vllm_kg_extraction(session, chunk)
                    else:
                        result = await self.test_ollama_kg_extraction(session, chunk)
                    
                    results.append(result)
                    
                    # Print quick results
                    if result.error:
                        print(f"ERROR - {result.error}")
                    else:
                        triples_count = len(result.extracted_triples) if result.extracted_triples else 0
                        print(f"{result.response_time:.2f}s ({result.tokens_per_second:.1f} tok/s, {triples_count} triples)")
                    
                    # Small delay between runs
                    await asyncio.sleep(2)
        
        return results
    
    async def run_kg_benchmark(self, text_chunks: List[str], runs_per_chunk: int = 2) -> Dict[str, List[KGBenchmarkResult]]:
        """Run knowledge graph extraction benchmark with services running one at a time"""
        print("🧠 Starting Knowledge Graph Extraction Benchmark")
        print(f"📊 Testing {len(text_chunks)} text chunks with {runs_per_chunk} runs each")
        print(f"📝 Using realistic txt2kg prompts for triple extraction")
        
        all_results = {}
        
        # First, stop all services to ensure clean start
        self.stop_all_services()
        
        # Test vLLM
        if self.start_vllm():
            vllm_results = await self.benchmark_service("vLLM", text_chunks, runs_per_chunk)
            all_results["vLLM"] = vllm_results
            self.stop_all_services()
        else:
            print("❌ Skipping vLLM benchmark due to startup failure")
            all_results["vLLM"] = []
        
        # Test Ollama
        if self.start_ollama():
            ollama_results = await self.benchmark_service("Ollama", text_chunks, runs_per_chunk)
            all_results["Ollama"] = ollama_results
            self.stop_all_services()
        else:
            print("❌ Skipping Ollama benchmark due to startup failure")
            all_results["Ollama"] = []
        
        return all_results
    
    def analyze_kg_results(self, results: Dict[str, List[KGBenchmarkResult]]):
        """Analyze and print knowledge graph extraction benchmark results"""
        print("\n" + "=" * 90)
        print("KNOWLEDGE GRAPH EXTRACTION BENCHMARK ANALYSIS")
        print("=" * 90)
        
        for service_name, service_results in results.items():
            print(f"\n{service_name} Results:")
            print("-" * 50)
            
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
            triple_counts = [len(r.extracted_triples) if r.extracted_triples else 0 for r in valid_results]
            
            print(f"Valid runs: {len(valid_results)}")
            print(f"Response time (avg): {statistics.mean(response_times):.3f}s")
            print(f"Response time (median): {statistics.median(response_times):.3f}s")
            print(f"Response time (min/max): {min(response_times):.3f}s / {max(response_times):.3f}s")
            print(f"Tokens/second (avg): {statistics.mean(tokens_per_second):.1f}")
            print(f"Tokens/second (median): {statistics.median(tokens_per_second):.1f}")
            print(f"Completion tokens (avg): {statistics.mean(completion_tokens):.1f}")
            print(f"Extracted triples (avg): {statistics.mean(triple_counts):.1f}")
            print(f"Extracted triples (median): {statistics.median(triple_counts):.1f}")
            print(f"Extracted triples (min/max): {min(triple_counts)} / {max(triple_counts)}")
            
            # Show sample extractions
            if valid_results:
                print(f"\nSample triple extraction:")
                sample_result = valid_results[0]
                if sample_result.extracted_triples:
                    for i, triple in enumerate(sample_result.extracted_triples[:3]):
                        print(f"  {i+1}. ({triple.get('subject', 'N/A')}, {triple.get('predicate', 'N/A')}, {triple.get('object', 'N/A')})")
                    if len(sample_result.extracted_triples) > 3:
                        print(f"  ... and {len(sample_result.extracted_triples) - 3} more")
        
        # Comparison
        vllm_valid = [r for r in results.get("vLLM", []) if not r.error]
        ollama_valid = [r for r in results.get("Ollama", []) if not r.error]
        
        if vllm_valid and ollama_valid:
            print("\n" + "=" * 50)
            print("KNOWLEDGE EXTRACTION PERFORMANCE COMPARISON")
            print("=" * 50)
            
            vllm_avg_response = statistics.mean([r.response_time for r in vllm_valid])
            ollama_avg_response = statistics.mean([r.response_time for r in ollama_valid])
            
            vllm_avg_tokens_sec = statistics.mean([r.tokens_per_second for r in vllm_valid])
            ollama_avg_tokens_sec = statistics.mean([r.tokens_per_second for r in ollama_valid])
            
            vllm_avg_triples = statistics.mean([len(r.extracted_triples) if r.extracted_triples else 0 for r in vllm_valid])
            ollama_avg_triples = statistics.mean([len(r.extracted_triples) if r.extracted_triples else 0 for r in ollama_valid])
            
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
            
            if vllm_avg_triples > ollama_avg_triples:
                extraction_ratio = vllm_avg_triples / ollama_avg_triples
                print(f"🧠 vLLM extracts {extraction_ratio:.2f}x MORE triples on average")
            else:
                extraction_ratio = ollama_avg_triples / vllm_avg_triples
                print(f"🧠 Ollama extracts {extraction_ratio:.2f}x MORE triples on average")

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Benchmark: vLLM vs Ollama")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per text chunk")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer chunks")
    
    args = parser.parse_args()
    
    benchmark = KGExtractionBenchmark()
    text_chunks = benchmark.get_realistic_text_chunks()
    
    if args.quick:
        text_chunks = text_chunks[:2]  # Use only first 2 chunks for quick test
    
    try:
        results = asyncio.run(benchmark.run_kg_benchmark(text_chunks, args.runs))
        benchmark.analyze_kg_results(results)
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user.")
        benchmark.stop_all_services()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        benchmark.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()
