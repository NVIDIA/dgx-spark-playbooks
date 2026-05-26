"""Library-level benchmark for the trained retrieval engine.

Measures throughput and latency of the HLLM retriever and optionally the
trained LightGBM re-ranker, in-process on GPU. There is no HTTP server, no
FastAPI, no JSON serialization — these numbers reflect what the hardware
can achieve for the recommendation workload, independent of any particular
serving stack.

Retrieval runs as `torch.mm + topk` on the first available CUDA device
(preferring GB300). Mathematically equivalent to FAISS IndexFlatIP — exact
inner-product search, no quantization — just executed on the GPU.

USAGE
  # Retrieval-only sweep
  uv run python assets/benchmark_retrieval.py

  # Retrieval + trained re-ranker
  uv run python assets/benchmark_retrieval.py --with-reranker

  # Custom sweep
  uv run python assets/benchmark_retrieval.py --users 1 100 10000 1000000

  # Save results to JSON
  uv run python assets/benchmark_retrieval.py --save bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from train_reranker_lightgbm import (
    FEATURE_COLS, build_user_samples, compute_item_stats, compute_user_stats,
)


class LGBMReranker:
    """LightGBM lambdarank re-ranker with GPU feature precompute.

    Vectorizes the per-pair feature build that train_reranker_lightgbm.py
    does in a Python loop, so 100K-user batches stay tractable in benchmark
    sweeps. Item-side features and user-side context are staged on GPU once;
    each scoring call gathers candidates, computes HLLM history sims, and
    submits a single chunked numpy batch to ``Booster.predict``.
    """

    HISTORY_RECENT_K = 10
    ITEM_FEATURE_COLS = [
        'item_total_purchases', 'item_unique_buyers',
        'item_pop_30d', 'item_pop_90d', 'item_pop_180d', 'item_trend',
        'item_recency_days', 'item_age_days',
        'log_price', 'title_length', 'desc_length', 'has_image',
    ]
    USER_SCALAR_COLS = [
        'user_total_purchases', 'user_unique_items',
        'user_avg_price', 'user_price_std', 'user_recency_days',
    ]

    def __init__(
        self,
        model_path: Path,
        processed_dir: Path,
        item_embeddings: np.ndarray,
        item_id_map: np.ndarray,
        device: torch.device,
    ) -> None:
        self.device = device
        self.booster = lgb.Booster(model_file=str(model_path))

        interactions = pd.read_parquet(processed_dir / 'dress_interactions.parquet')
        metadata = pd.read_parquet(processed_dir / 'dress_metadata.parquet')
        item_to_idx = {str(iid): i for i, iid in enumerate(item_id_map) if iid != '[PAD]'}
        item_idx_to_id = [str(i) for i in item_id_map]

        samples = build_user_samples(interactions, item_to_idx)
        if not samples:
            raise RuntimeError('No usable users for LightGBM reranker precompute.')
        self.n_users = len(samples)

        # Item-side feature lookup (n_items, 12) on GPU.
        item_stats = compute_item_stats(interactions, metadata)
        item_stats_arr = item_stats.reindex(item_idx_to_id).fillna(0)
        self._item_feat_gpu = torch.from_numpy(
            item_stats_arr[self.ITEM_FEATURE_COLS].to_numpy(dtype=np.float32),
        ).to(device)
        self._log_price_idx_in_item_block = self.ITEM_FEATURE_COLS.index('log_price')

        # User-side static info: padded recent-history embeddings + history idx
        # tensor (for is_repurchase) + scalar features.
        user_stats = compute_user_stats(samples, metadata, item_idx_to_id)
        dim = item_embeddings.shape[1]
        item_emb_t = torch.from_numpy(item_embeddings).to(device)

        max_hist = max(len(s[1]) for s in samples)
        self._hist_emb_padded = torch.zeros(
            self.n_users, self.HISTORY_RECENT_K, dim, dtype=torch.float32, device=device,
        )
        self._hist_mask = torch.zeros(
            self.n_users, self.HISTORY_RECENT_K, dtype=torch.bool, device=device,
        )
        self._history_idx = torch.full(
            (self.n_users, max_hist), -1, dtype=torch.long, device=device,
        )
        scalar_buf = np.zeros((self.n_users, len(self.USER_SCALAR_COLS)), dtype=np.float32)

        for i, (uid, hist_idxs, _, _) in enumerate(samples):
            recent = hist_idxs[-self.HISTORY_RECENT_K:]
            self._hist_emb_padded[i, :len(recent)] = item_emb_t[recent]
            self._hist_mask[i, :len(recent)] = True
            self._history_idx[i, :len(hist_idxs)] = torch.tensor(
                hist_idxs, dtype=torch.long, device=device,
            )
            us = user_stats[uid]
            for k, col in enumerate(self.USER_SCALAR_COLS):
                scalar_buf[i, k] = us[col]
        self._user_scalars_gpu = torch.from_numpy(scalar_buf).to(device)
        self._user_avg_price_idx = self.USER_SCALAR_COLS.index('user_avg_price')

        # Final feature column order must match training.
        self._feature_to_pos = {c: i for i, c in enumerate(FEATURE_COLS)}

    def _build_features_chunk(
        self,
        cand_idx: torch.Tensor,        # (n, K) int64 on device
        cand_scores: torch.Tensor,     # (n, K) float32 on device
        sample_user_idx: torch.Tensor, # (n,)   int64 on device
        item_emb_gpu: torch.Tensor,    # (n_items, dim)
    ) -> np.ndarray:
        n, K = cand_idx.shape
        K_full = len(FEATURE_COLS)
        device = self.device

        # Item-side block: (n, K, 12)
        item_feats = self._item_feat_gpu[cand_idx]

        # HLLM dot product is the retrieval score we already have.
        hllm_dot = cand_scores                                    # (n, K)

        # HLLM sims vs. recent history.
        hist_emb = self._hist_emb_padded[sample_user_idx]         # (n, hist_k, dim)
        hist_mask = self._hist_mask[sample_user_idx]              # (n, hist_k)
        cand_emb = item_emb_gpu[cand_idx]                         # (n, K, dim)
        sims = torch.bmm(cand_emb, hist_emb.transpose(1, 2))      # (n, K, hist_k)
        mask = hist_mask.unsqueeze(1)                             # (n, 1, hist_k)
        sims_masked_min = sims.masked_fill(~mask, -float('inf'))
        max_sim = sims_masked_min.max(dim=2).values
        sims_masked_zero = sims.masked_fill(~mask, 0.0)
        valid = mask.sum(dim=2).clamp(min=1).to(sims.dtype)
        avg_sim = sims_masked_zero.sum(dim=2) / valid

        # User scalar broadcast: (n, U) -> (n, K, U)
        user_block = self._user_scalars_gpu[sample_user_idx]
        user_block_kk = user_block.unsqueeze(1).expand(-1, K, -1)

        # is_repurchase: (n, K) bool
        hist_idx = self._history_idx[sample_user_idx]             # (n, max_hist)
        eq = cand_idx.unsqueeze(2) == hist_idx.unsqueeze(1)       # (n, K, max_hist)
        is_rep = eq.any(dim=2).to(torch.float32)

        # Cross features
        log_price = item_feats[..., self._log_price_idx_in_item_block]
        cand_price = torch.expm1(log_price)
        user_avg_price = user_block[:, self._user_avg_price_idx].unsqueeze(1)
        price_ratio = cand_price / (user_avg_price + 1e-8)
        price_diff = cand_price - user_avg_price

        # Assemble (n, K, 23) in FEATURE_COLS order.
        out = torch.empty(n, K, K_full, dtype=torch.float32, device=device)
        f2p = self._feature_to_pos
        out[..., f2p['hllm_dot_product']]    = hllm_dot
        out[..., f2p['hllm_max_hist_sim']]   = max_sim
        out[..., f2p['hllm_avg_hist_sim']]   = avg_sim
        for c, col in enumerate(self.ITEM_FEATURE_COLS):
            out[..., f2p[col]] = item_feats[..., c]
        for c, col in enumerate(self.USER_SCALAR_COLS):
            out[..., f2p[col]] = user_block_kk[..., c]
        out[..., f2p['price_ratio']] = price_ratio
        out[..., f2p['price_diff']]  = price_diff
        out[..., f2p['is_repurchase']] = is_rep

        return out.reshape(n * K, K_full).cpu().numpy()


def detect_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. This benchmark requires a CUDA-capable device "
            "(it characterizes the GPU's retrieval throughput on this hardware)."
        )
    # Prefer GB300 if multiple GPUs are present.
    for i in range(torch.cuda.device_count()):
        try:
            if 'GB300' in torch.cuda.get_device_name(i):
                return torch.device(f'cuda:{i}')
        except Exception:
            continue
    return torch.device('cuda:0')


class GpuSearcher:
    """torch.mm + topk on a CUDA device. Mathematically equivalent to FAISS IndexFlatIP
    (exact inner-product search, no quantization), just executed on the GPU.

    Chunks queries internally so the (queries × items) scores tensor stays under a
    fixed VRAM budget, allowing very large sweeps (e.g. 1M users) without OOM.
    """

    # Cap the on-GPU scores tensor so a 1M-user batch over a 16K-item index doesn't
    # try to allocate 64 GB at once. ~16 GiB gives plenty of headroom on GB300.
    _MAX_SCORES_BYTES = 16 * 1024**3

    def __init__(self, item_embeddings: np.ndarray, device: torch.device):
        self.device = device
        self.item_emb = torch.from_numpy(item_embeddings).to(device)
        gpu_name = torch.cuda.get_device_name(device)
        self.name = f"torch.mm + topk on {device} ({gpu_name})"
        n_items = item_embeddings.shape[0]
        self._chunk = max(1, self._MAX_SCORES_BYTES // (n_items * 4))

    def _search_one_torch(self, queries_gpu: torch.Tensor, k: int):
        """Internal: queries already on self.device. Returns GPU (distances, indices)."""
        scores = torch.mm(queries_gpu, self.item_emb.T)
        return torch.topk(scores, k, dim=1)

    def _search_one(self, queries_np: np.ndarray, k: int):
        q = torch.from_numpy(queries_np).to(self.device, non_blocking=True)
        return self._search_one_torch(q, k)

    def search(self, queries: np.ndarray, k: int):
        n_q = queries.shape[0]
        if n_q <= self._chunk:
            distances, indices = self._search_one(queries, k)
            torch.cuda.synchronize(self.device)
            return distances.cpu().numpy(), indices.cpu().numpy()

        all_d = np.empty((n_q, k), dtype=np.float32)
        all_i = np.empty((n_q, k), dtype=np.int64)
        for start in range(0, n_q, self._chunk):
            end = min(start + self._chunk, n_q)
            distances, indices = self._search_one(queries[start:end], k)
            all_d[start:end] = distances.cpu().numpy()
            all_i[start:end] = indices.cpu().numpy()
        torch.cuda.synchronize(self.device)
        return all_d, all_i

    def search_torch(self, queries: torch.Tensor, k: int):
        """All-GPU search: GPU torch in, GPU torch out. For end-to-end GPU pipelines
        where the reranker (or any downstream consumer) doesn't need a numpy round-trip."""
        if queries.device != self.device:
            queries = queries.to(self.device, non_blocking=True)
        n_q = queries.shape[0]
        if n_q <= self._chunk:
            d, i = self._search_one_torch(queries, k)
            torch.cuda.synchronize(self.device)
            return d, i

        all_d = torch.empty((n_q, k), dtype=torch.float32, device=self.device)
        all_i = torch.empty((n_q, k), dtype=torch.int64, device=self.device)
        for start in range(0, n_q, self._chunk):
            end = min(start + self._chunk, n_q)
            d, i = self._search_one_torch(queries[start:end], k)
            all_d[start:end] = d
            all_i[start:end] = i
        torch.cuda.synchronize(self.device)
        return all_d, all_i


def load_engine(processed_dir: Path, models_dir: Path, with_reranker: bool, device: torch.device):
    item_embeddings = np.load(processed_dir / 'hllm_item_embeddings.npy').astype(np.float32)
    item_id_map = np.load(processed_dir / 'hllm_item_id_map.npy', allow_pickle=True)
    interactions = pd.read_parquet(processed_dir / 'dress_interactions.parquet')
    interactions = interactions.sort_values(['user_id', 'timestamp'])

    item_to_idx = {str(iid): i for i, iid in enumerate(item_id_map) if iid != '[PAD]'}

    # Build a single (n_users, dim) matrix so we can sample with vectorized indexing
    # instead of a 1M-iteration Python lookup at very large sweep sizes.
    user_emb_list: list[np.ndarray] = []
    for _, group in interactions.groupby('user_id'):
        idxs = [item_to_idx[str(i)] for i in group['item_id'] if str(i) in item_to_idx]
        if idxs:
            emb = item_embeddings[idxs].mean(axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            user_emb_list.append(emb.astype(np.float32))
    user_emb_matrix = np.stack(user_emb_list)

    reranker = None
    user_matrix_gpu = None
    if with_reranker:
        lgbm_path = models_dir / 'reranker_lightgbm' / 'reranker_lightgbm.txt'
        if not lgbm_path.exists():
            raise FileNotFoundError(
                f"No re-ranker checkpoint found at {lgbm_path}\n"
                "Run Step 5 (`bash assets/train_reranker.sh`) first."
            )
        reranker = LGBMReranker(
            lgbm_path, processed_dir, item_embeddings, item_id_map, device,
        )
        user_matrix_gpu = torch.from_numpy(user_emb_matrix).to(device, non_blocking=True)

    return item_embeddings, user_emb_matrix, user_matrix_gpu, reranker


def score_candidates(
    reranker,
    item_emb_gpu: torch.Tensor,
    user_matrix_gpu: torch.Tensor,
    cand_idx_gpu: torch.Tensor,
    cand_scores_gpu: torch.Tensor | None = None,
    sample_user_idx_gpu: torch.Tensor | None = None,
    chunk_size: int = 2048,
) -> float:
    """Score (n_users × k_candidates) pairs with the LightGBM reranker and
    return elapsed seconds. GPU feature engineering, then chunked CPU
    ``Booster.predict``. ``cand_scores_gpu`` and ``sample_user_idx_gpu`` are
    required.
    """
    device = item_emb_gpu.device
    n_users, _k = cand_idx_gpu.shape
    if cand_idx_gpu.dtype != torch.int64:
        cand_idx_gpu = cand_idx_gpu.to(torch.int64)

    if cand_scores_gpu is None or sample_user_idx_gpu is None:
        raise ValueError('LGBMReranker requires cand_scores_gpu and sample_user_idx_gpu.')
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for start in range(0, n_users, chunk_size):
        end = min(start + chunk_size, n_users)
        features_np = reranker._build_features_chunk(
            cand_idx_gpu[start:end],
            cand_scores_gpu[start:end],
            sample_user_idx_gpu[start:end],
            item_emb_gpu,
        )
        _ = reranker.booster.predict(features_np)
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def bench_at_n(
    n: int,
    user_emb_matrix: np.ndarray,
    user_matrix_gpu: torch.Tensor | None,
    searcher: GpuSearcher,
    reranker,
    top_k: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_unique = user_emb_matrix.shape[0]
    has_rerank = reranker is not None
    device = searcher.device

    # Build the sample. Reranker path stays on GPU; retrieval-only path stays in numpy.
    # For LGBMReranker we also need the real-user index per query (history sims,
    # user scalars, is_repurchase). Track sample_user_idx_gpu always when reranking.
    sample_user_idx_gpu = None
    if has_rerank:
        if n <= n_unique:
            sample_user_idx_gpu = torch.arange(n, device=device, dtype=torch.long)
        else:
            sample_user_idx_gpu = torch.from_numpy(
                rng.integers(0, n_unique, size=n).astype(np.int64),
            ).to(device)
        sample_gpu = user_matrix_gpu[sample_user_idx_gpu]
    else:
        if n <= n_unique:
            sample = user_emb_matrix[:n]
        else:
            sample = user_emb_matrix[rng.integers(0, n_unique, size=n)]

    if n == 1:
        # Latency mode: 100 single-user iterations.
        if has_rerank:
            user_q_gpu = sample_gpu.reshape(1, -1)
            user_idx_q_gpu = sample_user_idx_gpu[:1]
            retr_lat, total_lat = [], []
            for _ in range(100):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                scores_gpu, idx_gpu = searcher.search_torch(user_q_gpu, top_k)
                torch.cuda.synchronize(device)
                retr_lat.append((time.perf_counter() - t0) * 1000)
                score_candidates(
                    reranker, searcher.item_emb, user_q_gpu, idx_gpu,
                    cand_scores_gpu=scores_gpu, sample_user_idx_gpu=user_idx_q_gpu,
                )
                total_lat.append((time.perf_counter() - t0) * 1000)
        else:
            user_emb = sample.reshape(1, -1)
            retr_lat = []
            for _ in range(100):
                t0 = time.perf_counter()
                _, _ = searcher.search(user_emb, top_k)
                retr_lat.append((time.perf_counter() - t0) * 1000)
            total_lat = retr_lat

        retr_ms = float(np.mean(retr_lat))
        total_ms = float(np.mean(total_lat))
        rerank_ms = total_ms - retr_ms
        return {
            'users': n,
            'retrieval_ms': retr_ms,
            'rerank_ms': rerank_ms,
            'total_ms': total_ms,
            'per_user_ms': total_ms,
            'throughput_rps': 1000.0 / total_ms,
        }

    # Batched mode: one shot.
    if has_rerank:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        scores_gpu, indices_gpu = searcher.search_torch(sample_gpu, top_k)
        torch.cuda.synchronize(device)
        retr_ms = (time.perf_counter() - t0) * 1000
        rerank_s = score_candidates(
            reranker, searcher.item_emb, sample_gpu, indices_gpu,
            cand_scores_gpu=scores_gpu, sample_user_idx_gpu=sample_user_idx_gpu,
        )
        rerank_ms = rerank_s * 1000
    else:
        t0 = time.perf_counter()
        _, _ = searcher.search(sample, top_k)
        retr_ms = (time.perf_counter() - t0) * 1000
        rerank_ms = 0.0

    total_ms = retr_ms + rerank_ms
    per_user_ms = total_ms / n
    return {
        'users': n,
        'retrieval_ms': retr_ms,
        'rerank_ms': rerank_ms,
        'total_ms': total_ms,
        'per_user_ms': per_user_ms,
        'throughput_rps': n * 1000.0 / total_ms,
    }


def _format_header() -> tuple[str, str]:
    header = f"{'Users':>11} | {'Per-user':>12} | {'Throughput':>14}"
    return header, "-" * len(header)


def _format_row(r: dict) -> str:
    return (
        f"{r['users']:>11,} | "
        f"{r['per_user_ms']:>10.3f}ms | "
        f"{r['throughput_rps']:>10,.0f} /s"
    )


def _fmt_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def main() -> int:
    workspace = Path(os.environ.get('PLAYBOOK_WORKSPACE', os.path.expanduser('~')))
    parser = argparse.ArgumentParser(description='In-process benchmark for the HLLM retrieval engine.')
    parser.add_argument('--processed-dir', default=str(workspace / 'data' / 'processed'))
    parser.add_argument('--models-dir', default=str(workspace / 'models'))
    parser.add_argument('--users', type=int, nargs='+',
                        default=[1, 1_000, 10_000, 100_000, 1_000_000],
                        help='User-batch sizes to sweep over (default: 1 1000 10000 100000 1000000).')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Retrieval depth — candidate set size before re-ranking (default: 100).')
    parser.add_argument('--with-reranker', action='store_true',
                        help='Also benchmark the trained LightGBM re-ranker over the retrieved candidates.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', help='Optional path to save full results as JSON.')
    args = parser.parse_args()

    try:
        device = detect_device()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("HLLM Retrieval Engine Benchmark")
    print("=" * 80)
    t = time.perf_counter()
    item_embeddings, user_emb_matrix, user_matrix_gpu, reranker = load_engine(
        Path(args.processed_dir), Path(args.models_dir), args.with_reranker, device,
    )
    searcher = GpuSearcher(item_embeddings, device)

    print(f"Loaded: {item_embeddings.shape[0]:,} items × {item_embeddings.shape[1]} dims, "
          f"{user_emb_matrix.shape[0]:,} user embeddings. "
          f"({time.perf_counter()-t:.1f}s)")
    print(f"Search backend: {searcher.name}")
    if reranker is not None:
        n_trees = reranker.booster.num_trees()
        print(f"Reranker: LightGBM ({n_trees} trees, {len(FEATURE_COLS)} features)")
    print(f"top_k retrieval depth: {args.top_k}")
    print()

    # Warm up the GPU once before the sweep so cuBLAS algorithm picks and CUDA
    # kernel JITs don't get charged to the first row's wall time.
    if reranker is not None:
        warmup_q_gpu = user_matrix_gpu[:1]
        warmup_idx = torch.zeros(1, dtype=torch.long, device=device)
        scores_gpu, idx_gpu = searcher.search_torch(warmup_q_gpu, args.top_k)
        _ = score_candidates(
            reranker, searcher.item_emb, warmup_q_gpu, idx_gpu,
            cand_scores_gpu=scores_gpu, sample_user_idx_gpu=warmup_idx,
        )
    else:
        _, _ = searcher.search(user_emb_matrix[:1], args.top_k)

    # Live progress: each row prints "Running X users..." before the work, then
    # "done (T)" once the batch completes. Users feel the wall time between the
    # two halves of the line.
    results = []
    for n in args.users:
        print(f"Running {n:>11,} users...", end=' ', flush=True)
        t0 = time.perf_counter()
        r = bench_at_n(n, user_emb_matrix, user_matrix_gpu, searcher, reranker,
                       args.top_k, args.seed)
        wall = time.perf_counter() - t0
        results.append(r)
        print(f"done ({_fmt_duration(wall)})", flush=True)

    # Final table — printed only once, after all benchmarks complete.
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    header, sep = _format_header()
    print(header)
    print(sep)
    for r in results:
        print(_format_row(r))

    if args.save:
        out = {
            'config': {
                'users': args.users,
                'top_k': args.top_k,
                'with_reranker': args.with_reranker,
                'search_backend': searcher.name,
                'item_count': int(item_embeddings.shape[0]),
                'embedding_dim': int(item_embeddings.shape[1]),
            },
            'results': results,
        }
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\nSaved JSON results to {args.save}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
