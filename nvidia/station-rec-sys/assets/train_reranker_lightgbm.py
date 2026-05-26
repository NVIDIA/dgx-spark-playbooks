"""Train a LightGBM lambdarank re-ranker on HLLM embeddings + handcrafted features.

Two-stage retrieval-and-rank pipeline: HLLM/FAISS produces top-100 candidates
per user, this script trains a LightGBM model to re-order them.

Pipeline per user (leave-last-out):
  1. Sort interactions by timestamp; hold out the last item as the positive.
  2. Build user_emb = mean(HLLM[history items]), L2-normalized.
  3. Retrieve top-100 candidates via torch.mm + topk over the item embedding
     matrix (mathematically equivalent to FAISS IndexFlatIP).
  4. If the held-out item lands in the top-100, label that row 1 and the
     other 99 rows 0 (a training-signal row group). Skip users whose
     positive missed the top-100 — there is nothing to learn for them.
  5. Engineer ~21 features per (user, candidate) pair: item popularity
     windows, user history stats, three HLLM similarity signals
     (dot-product, max/avg vs. recent history), price ratios, is_repurchase.

Train/valid: 80/20 split over user groups. LightGBM with the lambdarank
objective and group sizes = candidate-set size per user.

Inference contract: assets/app.py and assets/benchmark_retrieval.py load
the saved model and call model.predict(features) to score candidates.

Feature provenance: the handcrafted feature set is adapted from the
1st-place solution to the H&M Personalized Fashion Recommendations
Kaggle competition, which combined item popularity windows, user
history aggregates, price relationships, and pairwise text-similarity
signals into a LightGBM lambdarank model. We keep the H&M structure
and swap the original TF-IDF text similarities for HLLM embedding
similarities (`hllm_dot_product`, `hllm_max_hist_sim`, `hllm_avg_hist_sim`).

Generalization assumption: this feature set is expected to transfer to
other sparse retail datasets (long-tail item distributions, repeat-purchase
signal, mixed continuous + categorical + missing features). If you adapt
this to a different domain — content/video, travel, music, location-based
— expect to add domain-specific signals (watch time, location distance,
session co-occurrence, etc.) and possibly drop features that no longer
apply (e.g. `is_repurchase` is meaningless for one-time purchases).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch


def default_workspace() -> Path:
    return Path(os.environ.get('PLAYBOOK_WORKSPACE', os.path.expanduser('~')))


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_inputs(processed_dir: Path):
    interactions = pd.read_parquet(processed_dir / 'dress_interactions.parquet')
    metadata = pd.read_parquet(processed_dir / 'dress_metadata.parquet')
    embeddings = np.load(processed_dir / 'hllm_item_embeddings.npy').astype(np.float32)
    item_ids = np.load(processed_dir / 'hllm_item_id_map.npy', allow_pickle=True).astype(str)
    return interactions, metadata, embeddings, item_ids


def build_user_samples(interactions: pd.DataFrame, item_to_idx: dict[str, int]):
    """For each user with >=2 mapped interactions, emit (uid, history_idx, positive_idx, history_timestamps).

    history_idx is the user's interactions excluding the last; positive_idx is
    the last item. Returned order is users in interaction-frame order.
    """
    inter = interactions.sort_values(['user_id', 'timestamp'])
    samples = []
    for uid, group in inter.groupby('user_id', sort=False):
        item_seq = group['item_id'].tolist()
        ts_seq = group['timestamp'].tolist()
        idxs = [item_to_idx[i] for i in item_seq if i in item_to_idx]
        ts = [t for i, t in zip(item_seq, ts_seq) if i in item_to_idx]
        if len(idxs) < 2:
            continue
        samples.append((uid, idxs[:-1], idxs[-1], ts[:-1]))
    return samples


# ----------------------------------------------------------------------
# Candidate retrieval (FAISS-equivalent on GPU)
# ----------------------------------------------------------------------

def retrieve_candidates_gpu(
    user_emb_matrix: np.ndarray,
    item_emb_matrix: np.ndarray,
    top_k: int,
    device: torch.device,
    chunk_size: int = 4096,
):
    """torch.mm + topk on GPU. Chunked over users to bound peak memory."""
    item_t = torch.from_numpy(item_emb_matrix).to(device)
    n_users = user_emb_matrix.shape[0]
    out_idx = np.empty((n_users, top_k), dtype=np.int64)
    out_score = np.empty((n_users, top_k), dtype=np.float32)
    for start in range(0, n_users, chunk_size):
        end = min(start + chunk_size, n_users)
        u = torch.from_numpy(user_emb_matrix[start:end]).to(device)
        scores = torch.mm(u, item_t.T)
        top_s, top_i = torch.topk(scores, top_k, dim=1)
        out_idx[start:end] = top_i.cpu().numpy()
        out_score[start:end] = top_s.cpu().numpy()
    return out_idx, out_score


def build_user_embeddings(samples, embeddings: np.ndarray) -> np.ndarray:
    """L2-normalized mean of history-item embeddings per user."""
    n_users = len(samples)
    dim = embeddings.shape[1]
    user_emb = np.empty((n_users, dim), dtype=np.float32)
    for i, (_, hist_idxs, _, _) in enumerate(samples):
        emb = embeddings[hist_idxs].mean(axis=0)
        user_emb[i] = emb / (float(np.linalg.norm(emb)) + 1e-8)
    return user_emb


# ----------------------------------------------------------------------
# Item / user statistics for handcrafted features
# ----------------------------------------------------------------------

def compute_item_stats(interactions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-item rollups: total purchases, recency windows, trend, content."""
    max_ts = int(interactions['timestamp'].max())
    DAY = 86400
    win_30 = max_ts - 30 * DAY
    win_90 = max_ts - 90 * DAY
    win_180 = max_ts - 180 * DAY

    counts = interactions.groupby('item_id').agg(
        item_total_purchases=('user_id', 'size'),
        item_unique_buyers=('user_id', 'nunique'),
        item_first_seen=('timestamp', 'min'),
        item_last_seen=('timestamp', 'max'),
    )

    pop_30 = interactions[interactions['timestamp'] >= win_30]['item_id'].value_counts().rename('item_pop_30d')
    pop_90 = interactions[interactions['timestamp'] >= win_90]['item_id'].value_counts().rename('item_pop_90d')
    pop_180 = interactions[interactions['timestamp'] >= win_180]['item_id'].value_counts().rename('item_pop_180d')
    counts = counts.join(pop_30, how='left').join(pop_90, how='left').join(pop_180, how='left').fillna(0)

    counts['item_trend'] = (counts['item_pop_30d'] + 1) / (counts['item_pop_180d'] / 6 + 1)
    counts['item_age_days'] = (max_ts - counts['item_first_seen']) / DAY
    counts['item_recency_days'] = (max_ts - counts['item_last_seen']) / DAY
    counts = counts.drop(columns=['item_first_seen', 'item_last_seen'])

    meta = metadata.set_index('item_id').copy()
    meta['log_price'] = np.log1p(meta['price'].fillna(meta['price'].median()))
    meta['title_length'] = meta['title'].fillna('').str.len()
    meta['desc_length'] = meta['description'].fillna('').str.len()
    meta['has_image'] = (meta['image_url'].fillna('').str.len() > 0).astype(np.int8)
    meta = meta[['log_price', 'title_length', 'desc_length', 'has_image']]

    item_stats = counts.join(meta, how='left').fillna(0)
    return item_stats


def compute_user_stats(samples, metadata: pd.DataFrame, item_idx_to_id: list[str]) -> dict:
    """Per-user rollups derivable from history alone (no leakage)."""
    price_lookup = metadata.set_index('item_id')['price'].to_dict()
    DAY = 86400
    out = {}
    for uid, hist_idxs, _, ts in samples:
        prices = [
            float(price_lookup.get(item_idx_to_id[i], 0.0))
            for i in hist_idxs
        ]
        prices = [p for p in prices if p > 0]
        out[uid] = {
            'user_total_purchases': len(hist_idxs),
            'user_unique_items': len(set(hist_idxs)),
            'user_avg_price': float(np.mean(prices)) if prices else 0.0,
            'user_price_std': float(np.std(prices)) if len(prices) > 1 else 0.0,
            'user_recency_days': (max(ts) - min(ts)) / DAY if len(ts) > 1 else 0.0,
        }
    return out


# ----------------------------------------------------------------------
# Feature matrix construction
# ----------------------------------------------------------------------

FEATURE_COLS = [
    # HLLM signals (dominant; replace TF-IDF text similarity from H&M pipeline)
    'hllm_dot_product', 'hllm_max_hist_sim', 'hllm_avg_hist_sim',
    # Item popularity / lifecycle
    'item_total_purchases', 'item_unique_buyers',
    'item_pop_30d', 'item_pop_90d', 'item_pop_180d', 'item_trend',
    'item_recency_days', 'item_age_days',
    # Item content
    'log_price', 'title_length', 'desc_length', 'has_image',
    # User history
    'user_total_purchases', 'user_unique_items',
    'user_avg_price', 'user_price_std', 'user_recency_days',
    # Cross
    'price_ratio', 'price_diff', 'is_repurchase',
]


def build_feature_matrix(
    samples,
    cand_idx: np.ndarray,
    cand_scores: np.ndarray,
    embeddings: np.ndarray,
    item_idx_to_id: list[str],
    item_stats: pd.DataFrame,
    user_stats: dict,
    history_recent_k: int = 10,
):
    """Stack one row per (user, candidate) pair. Filters users where the
    held-out positive missed the candidate set.

    Returns:
        X: (n_rows, n_features) float32
        y: (n_rows,) int — 1 for the held-out positive, 0 otherwise
        groups: (n_kept_users,) int — group sizes for lambdarank
    """
    n_users, top_k = cand_idx.shape
    item_emb = embeddings  # (n_items, dim) numpy
    item_stats_arr = item_stats.reindex(item_idx_to_id).fillna(0)
    item_feat_lookup = item_stats_arr.to_numpy(dtype=np.float32)

    feature_to_col = {c: i for i, c in enumerate(FEATURE_COLS)}
    item_cols_in_stats = [
        'item_total_purchases', 'item_unique_buyers',
        'item_pop_30d', 'item_pop_90d', 'item_pop_180d', 'item_trend',
        'item_recency_days', 'item_age_days',
        'log_price', 'title_length', 'desc_length', 'has_image',
    ]
    stats_col_idx = [item_stats_arr.columns.get_loc(c) for c in item_cols_in_stats]

    rows_per_user = top_k
    X = np.zeros((n_users * rows_per_user, len(FEATURE_COLS)), dtype=np.float32)
    y = np.zeros(n_users * rows_per_user, dtype=np.int8)
    keep_user = np.zeros(n_users, dtype=bool)

    for u_i, (uid, hist_idxs, pos_idx, _) in enumerate(samples):
        cand = cand_idx[u_i]                             # (top_k,)
        scores = cand_scores[u_i]                        # (top_k,)
        if pos_idx not in cand:
            continue                                      # no signal for this user
        keep_user[u_i] = True

        row_start = u_i * rows_per_user
        row_end = row_start + rows_per_user

        # HLLM dot-product (already from torch.mm)
        X[row_start:row_end, feature_to_col['hllm_dot_product']] = scores

        # HLLM max / avg similarity vs. recent history
        recent_hist = hist_idxs[-history_recent_k:]
        hist_emb = item_emb[recent_hist]                  # (h, dim)
        cand_emb = item_emb[cand]                         # (top_k, dim)
        # Embeddings already L2-normalized at extraction time
        sim_matrix = cand_emb @ hist_emb.T                # (top_k, h)
        X[row_start:row_end, feature_to_col['hllm_max_hist_sim']] = sim_matrix.max(axis=1)
        X[row_start:row_end, feature_to_col['hllm_avg_hist_sim']] = sim_matrix.mean(axis=1)

        # Item-level features (vectorized lookup over candidates)
        for col_name, col_pos in zip(item_cols_in_stats, stats_col_idx):
            X[row_start:row_end, feature_to_col[col_name]] = item_feat_lookup[cand, col_pos]

        # User-level features (broadcast)
        u_stats = user_stats[uid]
        X[row_start:row_end, feature_to_col['user_total_purchases']] = u_stats['user_total_purchases']
        X[row_start:row_end, feature_to_col['user_unique_items']] = u_stats['user_unique_items']
        X[row_start:row_end, feature_to_col['user_avg_price']] = u_stats['user_avg_price']
        X[row_start:row_end, feature_to_col['user_price_std']] = u_stats['user_price_std']
        X[row_start:row_end, feature_to_col['user_recency_days']] = u_stats['user_recency_days']

        # Cross features
        cand_log_price_col = feature_to_col['log_price']
        cand_log_price = X[row_start:row_end, cand_log_price_col]
        cand_price = np.expm1(cand_log_price)
        u_avg_price = u_stats['user_avg_price']
        X[row_start:row_end, feature_to_col['price_ratio']] = cand_price / (u_avg_price + 1e-8)
        X[row_start:row_end, feature_to_col['price_diff']] = cand_price - u_avg_price

        hist_set = set(hist_idxs)
        X[row_start:row_end, feature_to_col['is_repurchase']] = np.array(
            [1.0 if int(c) in hist_set else 0.0 for c in cand], dtype=np.float32,
        )

        # Label
        for k, c in enumerate(cand):
            if int(c) == pos_idx:
                y[row_start + k] = 1
                break

    # Compact: drop dropped users' rows
    mask = np.repeat(keep_user, rows_per_user)
    X = X[mask]
    y = y[mask]
    groups = np.full(int(keep_user.sum()), rows_per_user, dtype=np.int64)
    return X, y, groups, int(keep_user.sum())


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    workspace = default_workspace()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('HLLM Re-ranker Training (LightGBM lambdarank)')
    print('=' * 60)
    print(f'  Processed dir: {processed_dir}')
    print(f'  Output dir:    {output_dir}')
    print(f'  Top-K:         {args.top_k}')
    print(f'  Boost rounds:  {args.num_rounds}')

    # ------ Load ------
    print('\n--- Loading data ---')
    t = time.time()
    interactions, metadata, embeddings, item_ids = load_inputs(processed_dir)
    item_to_idx = {iid: i for i, iid in enumerate(item_ids) if iid != '[PAD]'}
    item_idx_to_id = list(item_ids)
    print(f'  {len(interactions):,} interactions, '
          f'{embeddings.shape[0]:,} item embeddings ({embeddings.shape[1]} dim) in {time.time()-t:.1f}s')

    # ------ Build user samples (leave-last-out) ------
    print('\n--- Building user samples ---')
    t = time.time()
    samples = build_user_samples(interactions, item_to_idx)
    print(f'  Users with >=2 mapped interactions: {len(samples):,} ({time.time()-t:.1f}s)')

    # ------ User embeddings + GPU candidate retrieval ------
    print('\n--- Retrieving top-K candidates from HLLM embeddings ---')
    t = time.time()
    user_emb = build_user_embeddings(samples, embeddings)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cand_idx, cand_scores = retrieve_candidates_gpu(
        user_emb, embeddings, args.top_k, device,
    )
    print(f'  ({device}) Retrieved {cand_idx.shape[0]:,} × {args.top_k} '
          f'candidates in {time.time()-t:.1f}s')

    # ------ Statistics for handcrafted features ------
    print('\n--- Computing item and user statistics ---')
    t = time.time()
    item_stats = compute_item_stats(interactions, metadata)
    user_stats = compute_user_stats(samples, metadata, item_idx_to_id)
    print(f'  Item stats: {len(item_stats):,} rows × {len(item_stats.columns)} cols; '
          f'user stats: {len(user_stats):,} users ({time.time()-t:.1f}s)')

    # ------ Feature matrix ------
    print('\n--- Building feature matrix ---')
    t = time.time()
    X, y, groups, n_kept = build_feature_matrix(
        samples, cand_idx, cand_scores, embeddings,
        item_idx_to_id, item_stats, user_stats,
    )
    print(f'  Kept {n_kept:,}/{len(samples):,} users '
          f'(positive in top-{args.top_k}: {n_kept/len(samples):.1%}); '
          f'{X.shape[0]:,} rows × {X.shape[1]} features ({time.time()-t:.1f}s)')

    if n_kept < 100:
        raise RuntimeError(
            f'Only {n_kept} users have their held-out item in the top-{args.top_k} '
            'FAISS candidates. Retriever recall is too low to train a re-ranker. '
            'Train the retriever for more epochs or increase --top-k.'
        )

    # ------ Train/valid split (by user group) ------
    rng = np.random.default_rng(args.seed)
    n_groups = len(groups)
    perm = rng.permutation(n_groups)
    split = max(1, int(0.8 * n_groups))
    train_groups, valid_groups = perm[:split], perm[split:]

    rows_per_user = args.top_k
    train_row_mask = np.zeros(X.shape[0], dtype=bool)
    valid_row_mask = np.zeros(X.shape[0], dtype=bool)
    for gi in train_groups:
        train_row_mask[gi * rows_per_user:(gi + 1) * rows_per_user] = True
    for gi in valid_groups:
        valid_row_mask[gi * rows_per_user:(gi + 1) * rows_per_user] = True

    X_train, y_train = X[train_row_mask], y[train_row_mask]
    X_valid, y_valid = X[valid_row_mask], y[valid_row_mask]
    g_train = np.full(len(train_groups), rows_per_user, dtype=np.int64)
    g_valid = np.full(len(valid_groups), rows_per_user, dtype=np.int64)

    # ------ Train ------
    print(f'\n--- Training LightGBM lambdarank ---')
    print(f'  Train: {len(train_groups):,} users / {X_train.shape[0]:,} rows | '
          f'Valid: {len(valid_groups):,} users / {X_valid.shape[0]:,} rows')

    train_set = lgb.Dataset(X_train, y_train, group=g_train, feature_name=FEATURE_COLS)
    valid_set = lgb.Dataset(X_valid, y_valid, group=g_valid, feature_name=FEATURE_COLS,
                            reference=train_set)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],
        'lambdarank_truncation_level': args.lambdarank_truncation_level,
        'learning_rate': args.lr,
        'num_leaves': args.num_leaves,
        'max_depth': args.max_depth,
        'min_child_samples': 50,
        'min_gain_to_split': 0.0,
        'lambda_l1': args.lambda_l1,
        'lambda_l2': args.lambda_l2,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': args.seed,
    }
    if args.label_gain == 'binary':
        params['label_gain'] = [0, 1]

    eval_history: dict = {}
    t = time.time()
    model = lgb.train(
        params, train_set,
        num_boost_round=args.num_rounds,
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.record_evaluation(eval_history),
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
        ],
    )
    train_seconds = time.time() - t

    best_round = int(model.best_iteration)
    train_ndcg10 = float(eval_history['train']['ndcg@10'][best_round - 1]) if best_round else 0.0
    valid_ndcg10 = float(model.best_score['valid']['ndcg@10'])
    valid_ndcg5 = float(model.best_score['valid']['ndcg@5'])
    valid_ndcg20 = float(model.best_score['valid']['ndcg@20'])
    print(
        f'\nTrained {model.current_iteration()} rounds in {train_seconds:.1f}s '
        f'(best={best_round}, early-stopped at {args.early_stopping_rounds} rounds patience)'
    )
    print(
        f'  Best valid:  NDCG@5={valid_ndcg5:.4f}  NDCG@10={valid_ndcg10:.4f}  NDCG@20={valid_ndcg20:.4f}'
    )
    print(
        f'  Train/valid gap @10: {train_ndcg10 - valid_ndcg10:+.4f}  '
        f"(train NDCG@10={train_ndcg10:.4f}) — large positive = overfitting"
    )

    # ------ Feature importance ------
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')
    fi = sorted(
        zip(FEATURE_COLS, importance_gain, importance_split),
        key=lambda x: -x[1],
    )
    print('\nTop 10 features by gain:')
    for name, gain, split_count in fi[:10]:
        print(f'  {name:28s} gain={gain:>12,.0f}  splits={split_count:>5,}')

    # ------ Save ------
    model.save_model(str(output_dir / 'reranker_lightgbm.txt'))
    metrics = {
        'model': 'lightgbm_lambdarank',
        'feature_cols': FEATURE_COLS,
        'top_k': args.top_k,
        'num_rounds': args.num_rounds,
        'early_stopping_rounds': args.early_stopping_rounds,
        'best_iteration': best_round,
        'last_iteration': int(model.current_iteration()),
        'best_train_ndcg10': train_ndcg10,
        'best_valid_ndcg5':  valid_ndcg5,
        'best_valid_ndcg10': valid_ndcg10,
        'best_valid_ndcg20': valid_ndcg20,
        'train_valid_gap_ndcg10': train_ndcg10 - valid_ndcg10,
        'hyperparams': {
            'num_leaves': args.num_leaves,
            'learning_rate': args.lr,
            'lambda_l1': args.lambda_l1,
            'lambda_l2': args.lambda_l2,
            'feature_fraction': params['feature_fraction'],
            'bagging_fraction': params['bagging_fraction'],
        },
        'eval_history': {
            'train_ndcg10': [float(v) for v in eval_history['train']['ndcg@10']],
            'valid_ndcg10': [float(v) for v in eval_history['valid']['ndcg@10']],
            'valid_ndcg5':  [float(v) for v in eval_history['valid']['ndcg@5']],
            'valid_ndcg20': [float(v) for v in eval_history['valid']['ndcg@20']],
        },
        'n_users_trained': len(train_groups),
        'n_users_valid': len(valid_groups),
        'n_users_dropped': len(samples) - n_kept,
        'retriever_recall_at_top_k': n_kept / len(samples),
        'train_seconds': train_seconds,
        'feature_importance': [
            {'feature': n, 'gain': int(g), 'split': int(s)} for n, g, s in fi
        ],
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'\nSaved model to {output_dir / "reranker_lightgbm.txt"}')
    print(f'Saved metrics to {output_dir / "metrics.json"}')
    return metrics


def main() -> int:
    workspace = default_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', default=str(workspace / 'data' / 'processed'))
    parser.add_argument('--output-dir', default=str(workspace / 'models' / 'reranker_lightgbm'))
    parser.add_argument('--top-k', type=int, default=100,
                        help='Candidates per user from the HLLM retriever (default: 100).')
    parser.add_argument('--num-rounds', type=int, default=1000)
    parser.add_argument('--early-stopping-rounds', type=int, default=50)
    parser.add_argument('--num-leaves', type=int, default=63)
    parser.add_argument('--max-depth', type=int, default=8,
                        help='Tree depth cap. -1 disables. Defaults to 8 to constrain '
                             'overfitting on the small post-recall@K training set; see '
                             'docs/experiment-log.md ablation 2026-05-09.')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lambda-l1', type=float, default=0.0)
    parser.add_argument('--lambda-l2', type=float, default=1.0,
                        help='L2 regularization. Defaults to 1.0; see ablation 2026-05-09.')
    parser.add_argument('--lambdarank-truncation-level', type=int, default=30,
                        help='LightGBM default 30. Set near the eval NDCG cutoff for better head-of-list gradients.')
    parser.add_argument('--label-gain', choices=['default', 'binary'], default='default',
                        help='"default" = LightGBM graded-relevance gains; "binary" = [0,1] for 0/1 labels.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    t_total = time.time()
    train(args)
    print(f'\nTotal wall time: {time.time()-t_total:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
