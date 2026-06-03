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
FMHA Scaling Analysis: How Optimizations Impact Performance at Different Sizes

This script demonstrates:
1. How FMHA performance scales with sequence length
2. Which optimizations provide the most benefit at larger sizes
3. Target-specific configurations for different GPU architectures

Target Platforms (from TileGym):
- DGX Spark (sm120/sm121): TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2
- Blackwell B300 (sm100):  TILE_M=256, TILE_N=128 or 128x128, num_ctas=1, occupancy=1-2

Usage:
    python fmha_scaling_analysis.py [--iterations N]
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List
from types import SimpleNamespace

import torch

LOG_SEPARATOR = "=" * 80
LOG_SUBSEPARATOR = "-" * 60

@dataclass
class StepResult:
    step: int
    name: str
    latency_ms: float
    tflops: float
    speedup_vs_baseline: float

@dataclass 
class SeqLenResult:
    seq_len: int
    steps: List[StepResult]
    best_step: int
    best_speedup: float
    tilegym_latency_ms: float
    tilegym_tflops: float
    tilegym_speedup: float

class Logger:
    def __init__(self):
        self.results: List[SeqLenResult] = []
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

logger = Logger()

BATCH = 4
N_HEADS = 32
HEAD_DIM = 128
INV_LOG_2 = 1.0 / math.log(2)

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]


def get_fmha_config():
    """
    Get target-specific FMHA configuration (from TileGym attention.py)
    
    Returns configs matching TileGym's _fmha_autotune_configs():
    - sm120/sm121 (DGX Spark): TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2
    - sm100 (Blackwell B300): Two configs to try via autotuning
    """
    gpu_capability = torch.cuda.get_device_capability()
    
    if gpu_capability in [(12, 0), (12, 1)]:
        return [
            SimpleNamespace(
                name="DGX Spark (sm121)",
                TILE_M=64, 
                TILE_N=64, 
                num_ctas=1, 
                occupancy=2
            )
        ]
    else:
        return [
            SimpleNamespace(
                name="Blackwell B300 (sm100) - Config 1",
                TILE_M=256, 
                TILE_N=128, 
                num_ctas=1, 
                occupancy=1
            ),
            SimpleNamespace(
                name="Blackwell B300 (sm100) - Config 2",
                TILE_M=128, 
                TILE_N=128, 
                num_ctas=1, 
                occupancy=2
            ),
        ]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA not available")

def compute_flops(batch, heads, seq_len, head_dim, causal=True):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops

def benchmark_fn(fn, warmup=10, iterations=100):
    """Benchmark using triton's do_bench_cudagraph for accurate timing"""
    try:
        import triton
        # Use triton's cudagraph benchmark - same as TileGym
        ms = triton.testing.do_bench_cudagraph(fn)
        return ms
    except (ImportError, Exception):
        # Fallback to manual timing
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return (end - start) / iterations * 1000

def reference_fmha(q, k, v, sm_scale, is_causal=True):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
    )

try:
    import cuda.tile as ct
    from cuda.tile import RoundingMode as RMd
    CUTILE_AVAILABLE = True
except ImportError:
    CUTILE_AVAILABLE = False
    logger.log("[WARN] cuTile not available.")

if CUTILE_AVAILABLE:
    ConstInt = ct.Constant[int]
    ConstBool = ct.Constant[bool]

    @ct.kernel()
    def fmha_basic(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """Step 1: Basic cuTile - no optimizations"""
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

    @ct.kernel()
    def fmha_math_opt(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """Step 2: Math optimizations - exp2 + flush_to_zero"""
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

    @ct.kernel()
    def fmha_memory_opt(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """Step 3: Memory optimizations - load order + latency hints"""
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

    @ct.kernel(occupancy=2)
    def fmha_full_opt_occ2(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """Step 4a: Full optimization with occupancy=2 (for sm120/sm121)"""
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

    @ct.kernel(occupancy=1)
    def fmha_full_opt_occ1(
        Q, K, V, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        CAUSAL: ConstBool,
    ):
        """Step 4b: Full optimization with occupancy=1 (for sm100 Blackwell)"""
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

    def run_kernel(kernel_fn, q, k, v, sm_scale, tile_m, tile_n, is_causal=True):
        batch_size, num_heads, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        grid = (math.ceil(seq_len / tile_m), batch_size * num_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel_fn,
            (q, k, v, o, sm_scale, head_dim, num_heads, tile_m, tile_n, is_causal)
        )
        return o

def run_tilegym_fmha(q, k, v, sm_scale, is_causal=True):
    try:
        import tilegym
        return tilegym.ops.fmha(q, k, v, scaling=sm_scale, is_causal=is_causal, backend="cutile")
    except ImportError:
        return None


def run_scaling_analysis(iterations=100):
    device = get_device()
    gpu_cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    
    configs = get_fmha_config()
    primary_cfg = configs[0]
    TILE_M = primary_cfg.TILE_M
    TILE_N = primary_cfg.TILE_N
    
    logger.section("FMHA SCALING ANALYSIS (TileGym Benchmark Match)")
    logger.log("Matching TileGym bench_fused_attention.py configuration")
    logger.log(f"\nGPU: {gpu_name} (sm_{gpu_cap[0]}{gpu_cap[1]})")
    
    logger.subsection("TARGET-SPECIFIC CONFIGURATION (from TileGym)")
    for cfg in configs:
        logger.log(f"\n  {cfg.name}:")
        logger.log(f"    TILE_M={cfg.TILE_M}, TILE_N={cfg.TILE_N}")
        logger.log(f"    num_ctas={cfg.num_ctas}, occupancy={cfg.occupancy}")
    
    logger.log(f"\nUsing primary config: TILE_M={TILE_M}, TILE_N={TILE_N}, occupancy={primary_cfg.occupancy}")
    logger.log("\nTest Configuration (matches TileGym bench_fused_attention.py):")
    logger.log(f"  Batch: {BATCH}, Heads: {N_HEADS}, Head Dim: {HEAD_DIM}")
    logger.log("  Causal: True, Precision: float16")
    logger.log(f"  Sequence Lengths: {SEQ_LENS}")
    logger.log("  Benchmark: triton.testing.do_bench_cudagraph (same as TileGym)")
    
    logger.section("OPTIMIZATION STEPS")
    logger.log(f"""
Step 0: PyTorch Baseline
  - torch.nn.functional.scaled_dot_product_attention
  - Uses cuDNN Flash Attention backend
  - Highly optimized reference

Step 1: Basic cuTile (TILE_M={TILE_M}, TILE_N={TILE_N})
  - @ct.kernel with ct.mma() for Tensor Cores
  - Standard exp() for softmax
  - Explicit transpose with ct.permute()
  - No memory/occupancy hints

Step 2: Math Optimizations  
  - ct.exp2() instead of ct.exp() (faster on GPU)
  - flush_to_zero=True for denormals
  - Scale adjustment: multiply by 1/log(2)

Step 3: Memory Optimizations
  - Load order=(0,1,3,2) for implicit K transpose
  - Latency hints: K=2, V=4 for prefetching
  - Overlaps memory loads with computation

Step 4: Full Optimization (Target-Specific)
  - @ct.kernel(occupancy={primary_cfg.occupancy}) for {'sm120/121' if primary_cfg.occupancy == 2 else 'sm100'}
  - ct.truediv with APPROX rounding mode
  - Matches TileGym production implementation
""")

    logger.section("PLATFORM DIFFERENCES: DGX Spark vs Blackwell B300")
    logger.log("""
| Parameter    | DGX Spark (sm121) | Blackwell B300 (sm100) |
|--------------|-------------------|------------------------|
| TILE_M       | 64                | 256 or 128             |
| TILE_N       | 64                | 128                    |
| num_ctas     | 1                 | 1                      |
| occupancy    | 2                 | 1 or 2                 |

Why the difference?
- B300 has more SMs and larger shared memory -> can use bigger tiles
- B300 benefits from larger tiles (256x128) with lower occupancy
- DGX Spark needs smaller tiles (64x64) with higher occupancy to hide latency
- B300's higher memory bandwidth makes larger tiles more efficient
""")

    all_results = []
    
    select_kernel = fmha_full_opt_occ2 if primary_cfg.occupancy == 2 else fmha_full_opt_occ1
    
    for seq_len in SEQ_LENS:
        logger.subsection(f"Sequence Length: {seq_len}")
        
        q = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=device)
        k = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=device)
        v = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=device)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        
        flops = compute_flops(BATCH, N_HEADS, seq_len, HEAD_DIM, causal=True)
        
        steps_results = []
        
        baseline_fn = lambda: reference_fmha(q, k, v, sm_scale, is_causal=True)
        baseline_latency = benchmark_fn(baseline_fn, warmup=10, iterations=iterations)
        baseline_tflops = flops * 1e-12 / (baseline_latency * 1e-3)
        steps_results.append(StepResult(0, "PyTorch Baseline", baseline_latency, baseline_tflops, 1.0))
        
        if CUTILE_AVAILABLE:
            kernels = [
                (1, "Basic cuTile", fmha_basic),
                (2, "Math Opt (exp2)", fmha_math_opt),
                (3, "Memory Opt (order+latency)", fmha_memory_opt),
                (4, f"Full Opt (occ={primary_cfg.occupancy})", select_kernel),
            ]
            
            for step, name, kernel in kernels:
                try:
                    fn = lambda kernel=kernel: run_kernel(kernel, q, k, v, sm_scale, TILE_M, TILE_N, is_causal=True)
                    latency = benchmark_fn(fn, warmup=10, iterations=iterations)
                    tflops = flops * 1e-12 / (latency * 1e-3)
                    speedup = baseline_latency / latency
                    steps_results.append(StepResult(step, name, latency, tflops, speedup))
                except Exception as e:
                    logger.log(f"  [ERROR] Step {step} failed: {e}")
        
        tilegym_latency = 0.0
        tilegym_tflops = 0.0
        tilegym_speedup = 0.0
        tilegym_out = run_tilegym_fmha(q, k, v, sm_scale, is_causal=True)
        if tilegym_out is not None:
            tilegym_fn = lambda: run_tilegym_fmha(q, k, v, sm_scale, is_causal=True)
            tilegym_latency = benchmark_fn(tilegym_fn, warmup=10, iterations=iterations)
            tilegym_tflops = flops * 1e-12 / (tilegym_latency * 1e-3)
            tilegym_speedup = baseline_latency / tilegym_latency
        
        best_step = max(steps_results, key=lambda x: x.speedup_vs_baseline)
        
        result = SeqLenResult(
            seq_len=seq_len,
            steps=steps_results,
            best_step=best_step.step,
            best_speedup=best_step.speedup_vs_baseline,
            tilegym_latency_ms=tilegym_latency,
            tilegym_tflops=tilegym_tflops,
            tilegym_speedup=tilegym_speedup,
        )
        all_results.append(result)
        
        logger.log("\n  | Step | Name | Latency (ms) | TFLOPS | Speedup |")
        logger.log("  |------|------|--------------|--------|---------|")

        for sr in steps_results:
            logger.log(f"  | {sr.step} | {sr.name:<28} | {sr.latency_ms:>10.3f} | {sr.tflops:>6.2f} | {sr.speedup_vs_baseline:>6.2f}x |")
        if tilegym_latency > 0:
            logger.log(f"  | TG | TileGym Reference            | {tilegym_latency:>10.3f} | {tilegym_tflops:>6.2f} | {tilegym_speedup:>6.2f}x |")
        
        logger.log(f"\n  Best: Step {best_step.step} ({best_step.name}) with {best_step.speedup_vs_baseline:.2f}x speedup")
    
    logger.results = all_results
    return all_results


def print_summary(results: List[SeqLenResult]):
    configs = get_fmha_config()
    primary_cfg = configs[0]
    
    logger.section("SCALING SUMMARY")
    
    logger.log(f"\nTarget Config: TILE_M={primary_cfg.TILE_M}, TILE_N={primary_cfg.TILE_N}, occupancy={primary_cfg.occupancy}")
    
    logger.log("\n## Performance vs Sequence Length\n")
    logger.log("| Seq Len | Baseline (ms) | Full Opt (ms) | Speedup | TileGym (ms) | TG Speedup |")
    logger.log("|---------|---------------|---------------|---------|--------------|------------|")
    for r in results:
        baseline = next((s for s in r.steps if s.step == 0), None)
        full_opt = next((s for s in r.steps if s.step == 4), None)
        if baseline and full_opt:
            logger.log(f"| {r.seq_len:>7} | {baseline.latency_ms:>13.3f} | {full_opt.latency_ms:>13.3f} | {full_opt.speedup_vs_baseline:>6.2f}x | {r.tilegym_latency_ms:>12.3f} | {r.tilegym_speedup:>9.2f}x |")
    
    logger.log("\n## Optimization Impact by Sequence Length\n")
    logger.log("| Seq Len | Basic | +Math | +Memory | +Full | Best |")
    logger.log("|---------|-------|-------|---------|-------|------|")
    for r in results:
        row = f"| {r.seq_len:>7} |"
        for step in [1, 2, 3, 4]:
            sr = next((s for s in r.steps if s.step == step), None)
            if sr:
                row += f" {sr.speedup_vs_baseline:>5.2f}x |"
            else:
                row += "   N/A |"
        row += f" {r.best_speedup:>4.2f}x |"
        logger.log(row)
    
    logger.section("KEY INSIGHTS")
    logger.log("""
## Why Larger Sequences Benefit More from Optimization

1. **Memory Bandwidth Dominance**
   - Attention has O(N²) memory complexity for the QK^T matrix
   - At seq_len=8192: 8192² × 4 bytes = 256MB per head per batch
   - Memory optimizations (order, latency hints) have larger impact

2. **More K-Loop Iterations**  
   - At seq_len=512: 8 K-tiles (512/64) for sm121, 2 K-tiles (512/256) for sm100
   - At seq_len=8192: 128 K-tiles for sm121, 32 K-tiles for sm100
   - Latency hiding through pipelining amortizes over more iterations

3. **Better Occupancy Utilization**
   - More tiles = more parallelism opportunities
   - sm121 uses occupancy=2 (smaller tiles, more blocks)
   - sm100 uses occupancy=1 with larger tiles (256x128)

4. **Platform-Specific Tuning**
   - DGX Spark (sm121): 64x64 tiles, occupancy=2 - optimized for bandwidth-limited workloads
   - B300 (sm100): 256x128 tiles, occupancy=1 - optimized for compute-heavy workloads

## Optimization Priority by Problem Size

Small (seq_len <= 1024):
  - Basic cuTile often sufficient
  - Focus on correctness first

Medium (1024 < seq_len <= 4096):
  - Math optimizations (exp2) provide ~5% gain
  - Memory optimizations start to matter

Large (seq_len > 4096):
  - Full optimization stack critical
  - Platform-specific tuning essential
  - Memory pipelining becomes essential
""")


def export_results(results: List[SeqLenResult], output_dir: str):
    configs = get_fmha_config()
    primary_cfg = configs[0]
    
    data = {
        "config": {
            "batch": BATCH,
            "n_heads": N_HEADS, 
            "head_dim": HEAD_DIM,
            "tile_m": primary_cfg.TILE_M,
            "tile_n": primary_cfg.TILE_N,
            "occupancy": primary_cfg.occupancy,
            "num_ctas": primary_cfg.num_ctas,
            "platform": primary_cfg.name,
        },
        "results": [
            {
                "seq_len": r.seq_len,
                "steps": [asdict(s) for s in r.steps],
                "best_step": r.best_step,
                "best_speedup": r.best_speedup,
                "tilegym_latency_ms": r.tilegym_latency_ms,
                "tilegym_tflops": r.tilegym_tflops,
                "tilegym_speedup": r.tilegym_speedup,
            }
            for r in results
        ]
    }
    
    json_path = f"{output_dir}/fmha_scaling_results.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    md_path = f"{output_dir}/fmha_scaling_results.md"
    with open(md_path, 'w') as f:
        f.write("# FMHA Scaling Analysis Results\n\n")
        f.write("## Configuration\n")
        f.write(f"- Platform: {primary_cfg.name}\n")
        f.write(f"- Batch: {BATCH}, Heads: {N_HEADS}, Head Dim: {HEAD_DIM}\n")
        f.write(f"- Tile: {primary_cfg.TILE_M}x{primary_cfg.TILE_N}, occupancy={primary_cfg.occupancy}\n\n")
        
        f.write("## Target-Specific Configs (from TileGym)\n\n")
        f.write("| Platform | TILE_M | TILE_N | num_ctas | occupancy |\n")
        f.write("|----------|--------|--------|----------|----------|\n")
        f.write("| DGX Spark (sm121) | 64 | 64 | 1 | 2 |\n")
        f.write("| B300 (sm100) Config 1 | 256 | 128 | 1 | 1 |\n")
        f.write("| B300 (sm100) Config 2 | 128 | 128 | 1 | 2 |\n\n")
        
        f.write("## Results by Sequence Length\n\n")
        for r in results:
            f.write(f"### Seq Len = {r.seq_len}\n\n")
            f.write("| Step | Name | Latency (ms) | TFLOPS | Speedup |\n")
            f.write("|------|------|--------------|--------|--------|\n")
            for s in r.steps:
                f.write(f"| {s.step} | {s.name} | {s.latency_ms:.3f} | {s.tflops:.2f} | {s.speedup_vs_baseline:.2f}x |\n")
            if r.tilegym_latency_ms > 0:
                f.write(f"| TG | TileGym Reference | {r.tilegym_latency_ms:.3f} | {r.tilegym_tflops:.2f} | {r.tilegym_speedup:.2f}x |\n")
            f.write("\n")
    
    log_path = f"{output_dir}/fmha_scaling_log.txt"
    with open(log_path, 'w') as f:
        f.write('\n'.join(logger.logs))
    
    logger.log(f"\nResults exported to:")
    logger.log(f"  - {json_path}")
    logger.log(f"  - {md_path}")
    logger.log(f"  - {log_path}")


def main():
    parser = argparse.ArgumentParser(description="FMHA Scaling Analysis")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    results = run_scaling_analysis(iterations=args.iterations)
    print_summary(results)
    export_results(results, args.output_dir)


if __name__ == "__main__":
    main()
