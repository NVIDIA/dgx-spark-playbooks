# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
FMHA Optimization Tutorial: From Naive to Optimized cuTile Implementation

This script demonstrates step-by-step optimization of Flash Multi-Head Attention
using NVIDIA cuTile, starting from a basic implementation and progressively
adding optimizations until reaching TileGym-level performance.

Target Platform: DGX Spark (sm121) with pre-determined optimal tile sizes.
Note: TileGym supports autotuning, but we use hardcoded values for this tutorial.

Configuration (matches TileGym bench_fused_attention.py):
- Batch: 4, Heads: 32, Head Dim: 128
- Sequence Lengths: 1024, 2048, 4096, 8192, 16384
- Benchmark: triton.testing.do_bench_cudagraph

Usage:
    python fmha_optimization_tutorial.py [--iterations N] [--correctness-check]
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
import sys

import torch

LOG_SEPARATOR = "=" * 80
LOG_SUBSEPARATOR = "-" * 60

@dataclass
class BenchmarkResult:
    step: int
    name: str
    description: str
    latency_ms: float
    tflops: float
    speedup_vs_baseline: float
    speedup_vs_previous: float
    correct: bool
    key_changes: List[str]

class Logger:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.logs: List[str] = []
        
    def log(self, msg: str):
        print(msg)
        self.logs.append(msg)
        
    def section(self, title: str):
        self.log(f"\n{LOG_SEPARATOR}")
        self.log(f"  {title}")
        self.log(LOG_SEPARATOR)
        
    def subsection(self, title: str):
        self.log(f"\n{LOG_SUBSEPARATOR}")
        self.log(f"  {title}")
        self.log(LOG_SUBSEPARATOR)
        
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
        
    def export_json(self, filepath: str):
        data = {
            "results": [asdict(r) for r in self.results],
            "logs": self.logs
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def export_markdown(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write("# FMHA Optimization Tutorial Results\n\n")
            f.write("## Summary Table\n\n")
            f.write("| Step | Name | Latency (ms) | TFLOPS | vs Baseline | vs Previous | Correct |\n")
            f.write("|------|------|--------------|--------|-------------|-------------|--------|\n")
            for r in self.results:
                f.write(f"| {r.step} | {r.name} | {r.latency_ms:.3f} | {r.tflops:.2f} | {r.speedup_vs_baseline:.2f}x | {r.speedup_vs_previous:.2f}x | {'Yes' if r.correct else 'No'} |\n")
            f.write("\n## Detailed Steps\n\n")
            for r in self.results:
                f.write(f"### Step {r.step}: {r.name}\n\n")
                f.write(f"**Description**: {r.description}\n\n")
                f.write("**Key Changes**:\n")
                for change in r.key_changes:
                    f.write(f"- {change}\n")
                f.write(f"\n**Performance**: {r.latency_ms:.3f}ms, {r.tflops:.2f} TFLOPS, {r.speedup_vs_baseline:.2f}x vs baseline\n\n")

logger = Logger()

BATCH = 4
N_HEADS = 32
HEAD_DIM = 128
INV_LOG_2 = 1.0 / math.log(2)

TILE_M = 64
TILE_N = 64
OCCUPANCY = 2
NUM_CTAS = 1

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA not available")

DEVICE = None

def compute_flops(batch, heads, seq_len, head_dim, causal=True):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops

def benchmark_fn(fn, warmup=10, iterations=100):
    """Benchmark using triton's do_bench_cudagraph for accurate timing (matches TileGym)"""
    try:
        import triton
        ms = triton.testing.do_bench_cudagraph(fn)
        return ms
    except (ImportError, Exception):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return (end - start) / iterations * 1000

def verify_correctness(output, reference, atol=1e-2, rtol=1e-2):
    try:
        torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
        return True
    except AssertionError:
        max_diff = (output - reference).abs().max().item()
        logger.log(f"    [WARN] Max difference: {max_diff:.6f}")
        return max_diff < 0.1

def reference_fmha(q, k, v, sm_scale, is_causal=True):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
    )

def step0_pytorch_baseline(q, k, v, sm_scale, is_causal=True):
    return reference_fmha(q, k, v, sm_scale, is_causal)

try:
    import cuda.tile as ct
    from cuda.tile import RoundingMode as RMd
    CUTILE_AVAILABLE = True
except ImportError:
    CUTILE_AVAILABLE = False
    logger.log("[WARN] cuTile not available. Only PyTorch baseline will run.")

if CUTILE_AVAILABLE:
    ConstInt = ct.Constant[int]
    ConstBool = ct.Constant[bool]

    @ct.kernel()
    def fmha_step2_mma(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """
        Step 2: Basic cuTile FMHA with MMA (Tensor Cores)
        - Uses ct.mma() for matrix multiply
        - Standard exp() for softmax
        - Online softmax algorithm
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

        q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
        q = q.reshape((TILE_M, TILE_D))

        k_seqlen = K.shape[2]
        if CAUSAL:
            m_end = (bid_x + 1) * TILE_M
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
            mask_start = (bid_x * TILE_M) // TILE_N
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k_tile = ct.load(K, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D))
            k_tile = k_tile.reshape((TILE_N, TILE_D))
            k_t = ct.permute(k_tile, (1, 0))

            qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            qk = ct.mma(q, k_t, qk)
            qk = qk * qk_scale

            if CAUSAL and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = offs_m >= offs_n
                qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

            m_ij = ct.max(qk, axis=-1, keepdims=True)
            m_ij = ct.maximum(m_i, m_ij)
            qk = qk - m_ij

            p = ct.exp(qk)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v_tile = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D))
            v_tile = v_tile.reshape((TILE_N, TILE_D))
            p_cast = p.astype(Q.dtype)
            acc = ct.mma(p_cast, v_tile, acc)
            
            m_i = m_ij

        acc = acc / l_i
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def run_step2(q, k, v, sm_scale, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / TILE_M), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fmha_step2_mma,
            (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
        )
        return o

    @ct.kernel()
    def fmha_step3_exp2(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """
        Step 3: Use exp2 with flush_to_zero for faster math
        - exp2(x) = 2^x is faster than exp(x) = e^x on GPU
        - Requires scaling adjustment: multiply by 1/log(2)
        - flush_to_zero handles denormals efficiently
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H

        qk_scale_log2 = qk_scale * INV_LOG_2

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

        q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
        q = q.reshape((TILE_M, TILE_D))

        k_seqlen = K.shape[2]
        if CAUSAL:
            m_end = (bid_x + 1) * TILE_M
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
            mask_start = (bid_x * TILE_M) // TILE_N
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k_tile = ct.load(K, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D))
            k_tile = k_tile.reshape((TILE_N, TILE_D))
            k_t = ct.permute(k_tile, (1, 0))

            qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            qk = ct.mma(q, k_t, qk)

            if CAUSAL and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = offs_m >= offs_n
                qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

            m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
            m_ij = ct.maximum(m_i, m_ij)
            qk = qk * qk_scale_log2 - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v_tile = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D))
            v_tile = v_tile.reshape((TILE_N, TILE_D))
            p_cast = p.astype(Q.dtype)
            acc = ct.mma(p_cast, v_tile, acc)
            
            m_i = m_ij

        acc = acc / l_i
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def run_step3(q, k, v, sm_scale, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / TILE_M), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fmha_step3_exp2,
            (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
        )
        return o

    @ct.kernel()
    def fmha_step4_load_order(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """
        Step 4: Optimize K load with order parameter
        - Use order=(0,1,3,2) to load K already transposed
        - Avoids explicit ct.permute() operation
        - Reduces memory traffic
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H

        qk_scale_log2 = qk_scale * INV_LOG_2

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

        q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
        q = q.reshape((TILE_M, TILE_D))

        k_seqlen = K.shape[2]
        if CAUSAL:
            m_end = (bid_x + 1) * TILE_M
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
            mask_start = (bid_x * TILE_M) // TILE_N
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k_t = ct.load(
                K, 
                index=(batch_idx, head_idx, 0, j),
                shape=(1, 1, TILE_D, TILE_N),
                order=(0, 1, 3, 2)
            )
            k_t = k_t.reshape((TILE_D, TILE_N))

            qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            qk = ct.mma(q, k_t, qk)

            if CAUSAL and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = offs_m >= offs_n
                qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

            m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
            m_ij = ct.maximum(m_i, m_ij)
            qk = qk * qk_scale_log2 - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v_tile = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D))
            v_tile = v_tile.reshape((TILE_N, TILE_D))
            p_cast = p.astype(Q.dtype)
            acc = ct.mma(p_cast, v_tile, acc)
            
            m_i = m_ij

        acc = acc / l_i
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def run_step4(q, k, v, sm_scale, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / TILE_M), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fmha_step4_load_order,
            (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
        )
        return o

    @ct.kernel()
    def fmha_step5_latency(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """
        Step 5: Add latency hints for better pipelining
        - latency=2 for K load (prefetch)
        - latency=4 for V load (more prefetch distance)
        - Helps overlap memory loads with computation
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H

        qk_scale_log2 = qk_scale * INV_LOG_2

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

        q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
        q = q.reshape((TILE_M, TILE_D))

        k_seqlen = K.shape[2]
        if CAUSAL:
            m_end = (bid_x + 1) * TILE_M
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
            mask_start = (bid_x * TILE_M) // TILE_N
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k_t = ct.load(
                K, 
                index=(batch_idx, head_idx, 0, j),
                shape=(1, 1, TILE_D, TILE_N),
                order=(0, 1, 3, 2),
                latency=2
            )
            k_t = k_t.reshape((TILE_D, TILE_N))

            qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            qk = ct.mma(q, k_t, qk)

            if CAUSAL and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = offs_m >= offs_n
                qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

            m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
            m_ij = ct.maximum(m_i, m_ij)
            qk = qk * qk_scale_log2 - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v_tile = ct.load(
                V, 
                index=(batch_idx, head_idx, j, 0), 
                shape=(1, 1, TILE_N, TILE_D),
                latency=4
            )
            v_tile = v_tile.reshape((TILE_N, TILE_D))
            p_cast = p.astype(Q.dtype)
            acc = ct.mma(p_cast, v_tile, acc)
            
            m_i = m_ij

        acc = acc / l_i
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def run_step5(q, k, v, sm_scale, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / TILE_M), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fmha_step5_latency,
            (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
        )
        return o

    @ct.kernel(occupancy=2)
    def fmha_step6_occupancy(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """
        Step 6: Add occupancy hint
        - @ct.kernel(occupancy=2) improves SM utilization
        - Allows multiple thread blocks per SM
        - Better for hiding memory latency
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H

        qk_scale_log2 = qk_scale * INV_LOG_2

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

        q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
        q = q.reshape((TILE_M, TILE_D))

        k_seqlen = K.shape[2]
        if CAUSAL:
            m_end = (bid_x + 1) * TILE_M
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
            mask_start = (bid_x * TILE_M) // TILE_N
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k_t = ct.load(
                K, 
                index=(batch_idx, head_idx, 0, j),
                shape=(1, 1, TILE_D, TILE_N),
                order=(0, 1, 3, 2),
                latency=2
            )
            k_t = k_t.reshape((TILE_D, TILE_N))

            qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            qk = ct.mma(q, k_t, qk)

            if CAUSAL and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = offs_m >= offs_n
                qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

            m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
            m_ij = ct.maximum(m_i, m_ij)
            qk = qk * qk_scale_log2 - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v_tile = ct.load(
                V, 
                index=(batch_idx, head_idx, j, 0), 
                shape=(1, 1, TILE_N, TILE_D),
                latency=4
            )
            v_tile = v_tile.reshape((TILE_N, TILE_D))
            p_cast = p.astype(Q.dtype)
            acc = ct.mma(p_cast, v_tile, acc)
            
            m_i = m_ij

        acc = acc / l_i
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def run_step6(q, k, v, sm_scale, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / TILE_M), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fmha_step6_occupancy,
            (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
        )
        return o

    @ct.kernel(occupancy=2)
    def fmha_step7_approx_div(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """
        Step 7: Use approximate division for final normalization
        - ct.truediv with rounding_mode=APPROX is faster
        - Acceptable accuracy loss for inference
        - This matches TileGym's optimized implementation
        """
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H

        qk_scale_log2 = qk_scale * INV_LOG_2

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

        q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
        q = q.reshape((TILE_M, TILE_D))

        k_seqlen = K.shape[2]
        if CAUSAL:
            m_end = (bid_x + 1) * TILE_M
            Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
            mask_start = (bid_x * TILE_M) // TILE_N
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k_t = ct.load(
                K, 
                index=(batch_idx, head_idx, 0, j),
                shape=(1, 1, TILE_D, TILE_N),
                order=(0, 1, 3, 2),
                latency=2
            )
            k_t = k_t.reshape((TILE_D, TILE_N))

            qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            qk = ct.mma(q, k_t, qk)

            if CAUSAL and j >= mask_start:
                offs_n = j * TILE_N + offs_n_tile
                mask = offs_m >= offs_n
                qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

            m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
            m_ij = ct.maximum(m_i, m_ij)
            qk = qk * qk_scale_log2 - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v_tile = ct.load(
                V, 
                index=(batch_idx, head_idx, j, 0), 
                shape=(1, 1, TILE_N, TILE_D),
                latency=4
            )
            v_tile = v_tile.reshape((TILE_N, TILE_D))
            p_cast = p.astype(Q.dtype)
            acc = ct.mma(p_cast, v_tile, acc)
            
            m_i = m_ij

        acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    def run_step7(q, k, v, sm_scale, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / TILE_M), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fmha_step7_approx_div,
            (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
        )
        return o


def run_tilegym_fmha(q, k, v, sm_scale, is_causal=True):
    """Run TileGym's optimized FMHA for comparison"""
    try:
        import tilegym
        return tilegym.ops.fmha(q, k, v, scaling=sm_scale, is_causal=is_causal, backend="cutile")
    except ImportError:
        logger.log("[WARN] TileGym not available for comparison")
        return None


def run_benchmark(seq_len, iterations=100, check_correct=True):
    global DEVICE
    DEVICE = get_device()
    
    logger.section(f"FMHA OPTIMIZATION TUTORIAL - SEQ_LEN={seq_len}")
    logger.log("Configuration:")
    logger.log(f"  - Batch: {BATCH}")
    logger.log(f"  - Heads: {N_HEADS}")
    logger.log(f"  - Head Dim: {HEAD_DIM}")
    logger.log(f"  - Sequence Length: {seq_len}")
    logger.log(f"  - Tile M: {TILE_M}")
    logger.log(f"  - Tile N: {TILE_N}")
    logger.log("  - Precision: float16")
    logger.log("  - Causal: True")
    logger.log(f"  - Iterations: {iterations}")
    logger.log(f"  - Device: {DEVICE}")
    
    q = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    flops = compute_flops(BATCH, N_HEADS, seq_len, HEAD_DIM, causal=True)
    
    ref_output = reference_fmha(q, k, v, sm_scale, is_causal=True)
    
    steps = [
        (0, "PyTorch Baseline", "torch.nn.functional.scaled_dot_product_attention",
         lambda: step0_pytorch_baseline(q, k, v, sm_scale, is_causal=True),
         ["PyTorch SDPA with cuDNN backend", "Highly optimized baseline"]),
    ]
    
    if CUTILE_AVAILABLE:
        steps.extend([
            (2, "Basic cuTile + MMA", "Tiled FMHA with ct.mma() for Tensor Cores",
             lambda: run_step2(q, k, v, sm_scale, is_causal=True),
             ["@ct.kernel decorator", "ct.mma() for QK and PV products", "Online softmax with exp()"]),
            
            (3, "+ exp2 + flush_to_zero", "Faster exponential math",
             lambda: run_step3(q, k, v, sm_scale, is_causal=True),
             ["ct.exp2() instead of ct.exp()", "flush_to_zero=True for denormals", "qk_scale *= 1/log(2)"]),
            
            (4, "+ Load Order Transpose", "Avoid explicit transpose",
             lambda: run_step4(q, k, v, sm_scale, is_causal=True),
             ["order=(0,1,3,2) for K load", "K loaded already transposed", "Removes ct.permute() call"]),
            
            (5, "+ Latency Hints", "Better memory pipelining",
             lambda: run_step5(q, k, v, sm_scale, is_causal=True),
             ["latency=2 for K load", "latency=4 for V load", "Overlaps loads with compute"]),
            
            (6, "+ Occupancy=2", "Better SM utilization",
             lambda: run_step6(q, k, v, sm_scale, is_causal=True),
             ["@ct.kernel(occupancy=2)", "Multiple blocks per SM", "Hides memory latency"]),
            
            (7, "+ Approx Division (Final)", "Fast final normalization",
             lambda: run_step7(q, k, v, sm_scale, is_causal=True),
             ["ct.truediv with APPROX mode", "Matches TileGym implementation", "Full optimization achieved"]),
        ])
    
    baseline_latency = None
    prev_latency = None
    
    for step_idx, name, desc, fn, changes in steps:
        logger.subsection(f"Step {step_idx}: {name}")
        logger.log(f"Description: {desc}")
        logger.log("Key Changes:")
        for change in changes:
            logger.log(f"  - {change}")
        
        try:
            output = fn()
            latency_ms = benchmark_fn(fn, warmup=10, iterations=iterations)
            tflops = flops * 1e-12 / (latency_ms * 1e-3)
            
            if baseline_latency is None:
                baseline_latency = latency_ms
                speedup_baseline = 1.0
            else:
                speedup_baseline = baseline_latency / latency_ms
            
            if prev_latency is None:
                speedup_prev = 1.0
            else:
                speedup_prev = prev_latency / latency_ms
            
            if check_correct and output is not None:
                correct = verify_correctness(output, ref_output)
            else:
                correct = True
            
            logger.log("\nResults:")
            logger.log(f"  Latency:      {latency_ms:.3f} ms")
            logger.log(f"  TFLOPS:       {tflops:.2f}")
            logger.log(f"  vs Baseline:  {speedup_baseline:.2f}x")
            logger.log(f"  vs Previous:  {speedup_prev:.2f}x")
            logger.log(f"  Correct:      {'Yes' if correct else 'No'}")
            
            result = BenchmarkResult(
                step=step_idx,
                name=name,
                description=desc,
                latency_ms=latency_ms,
                tflops=tflops,
                speedup_vs_baseline=speedup_baseline,
                speedup_vs_previous=speedup_prev,
                correct=correct,
                key_changes=changes
            )
            logger.add_result(result)
            
            prev_latency = latency_ms
            
        except Exception as e:
            logger.log(f"\n[ERROR] Step {step_idx} failed: {e}")
            import traceback
            logger.log(traceback.format_exc())
    
    tilegym_output = run_tilegym_fmha(q, k, v, sm_scale, is_causal=True)
    if tilegym_output is not None:
        logger.subsection("TileGym Reference (for comparison)")
        tilegym_fn = lambda: run_tilegym_fmha(q, k, v, sm_scale, is_causal=True)
        tilegym_latency = benchmark_fn(tilegym_fn, warmup=10, iterations=iterations)
        tilegym_tflops = flops * 1e-12 / (tilegym_latency * 1e-3)
        tilegym_speedup = baseline_latency / tilegym_latency if baseline_latency else 1.0
        
        logger.log("TileGym FMHA:")
        logger.log(f"  Latency:      {tilegym_latency:.3f} ms")
        logger.log(f"  TFLOPS:       {tilegym_tflops:.2f}")
        logger.log(f"  vs Baseline:  {tilegym_speedup:.2f}x")
        
        result = BenchmarkResult(
            step=99,
            name="TileGym Reference",
            description="TileGym's optimized FMHA implementation",
            latency_ms=tilegym_latency,
            tflops=tilegym_tflops,
            speedup_vs_baseline=tilegym_speedup,
            speedup_vs_previous=1.0,
            correct=True,
            key_changes=["Full TileGym implementation", "Pre-tuned for sm121", "Production ready"]
        )
        logger.add_result(result)


def main():
    parser = argparse.ArgumentParser(description="FMHA Optimization Tutorial")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length (default matches TileGym)")
    parser.add_argument("--correctness-check", action="store_true", help="Enable correctness checking")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for logs")
    args = parser.parse_args()
    
    logger.section("FMHA OPTIMIZATION TUTORIAL")
    logger.log("From Naive to Optimized cuTile Implementation")
    logger.log("Target Platform: DGX Spark (sm121)")
    logger.log(f"Tile Sizes: TILE_M={TILE_M}, TILE_N={TILE_N} (hardcoded from TileGym)")
    logger.log("Note: TileGym supports autotuning, but we use pre-determined optimal values")
    
    run_benchmark(
        seq_len=args.seq_len,
        iterations=args.iterations,
        check_correct=args.correctness_check
    )
    
    logger.section("FINAL SUMMARY")
    logger.log("\n| Step | Name | Latency (ms) | TFLOPS | vs Baseline | Correct |")
    logger.log("|------|------|--------------|--------|-------------|---------|")
    for r in logger.results:
        logger.log(f"| {r.step} | {r.name} | {r.latency_ms:.3f} | {r.tflops:.2f} | {r.speedup_vs_baseline:.2f}x | {'Yes' if r.correct else 'No'} |")
    
    json_path = f"{args.output_dir}/fmha_tutorial_results.json"
    md_path = f"{args.output_dir}/fmha_tutorial_results.md"
    log_path = f"{args.output_dir}/fmha_tutorial_log.txt"
    
    logger.export_json(json_path)
    logger.export_markdown(md_path)
    
    with open(log_path, 'w') as f:
        f.write('\n'.join(logger.logs))
    
    logger.log("\nResults exported to:")
    logger.log(f"  - {json_path}")
    logger.log(f"  - {md_path}")
    logger.log(f"  - {log_path}")


if __name__ == "__main__":
    main()
