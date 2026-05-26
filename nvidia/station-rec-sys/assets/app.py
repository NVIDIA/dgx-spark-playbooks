"""Enterprise Fashion Recommender — FastAPI + vanilla HTML/CSS

Serves real HLLM-trained recommendations via FAISS nearest-neighbor search,
re-ranked by the LightGBM lambdarank model trained in Step 5. Item prices
shown in the UI are the PPO pricing agent's optimized prices by default;
pass --static-prices to display the original catalog (MSRP) prices instead.

Default:           python app.py                  (HLLM retrieval + LightGBM rerank, PPO-optimized prices)
Retriever only:    python app.py --retriever-only (HLLM retrieval, no rerank — for ablation)
Static prices:     python app.py --static-prices  (display original MSRP instead of PPO-optimized prices)

Opens at http://localhost:7860
"""

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from prepare_data import looks_non_dress


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    ap.add_argument('--retriever-only', action='store_true',
                    help='Skip the LightGBM re-ranker. Returns FAISS top-K directly.')
    ap.add_argument('--static-prices', action='store_true',
                    help='Show the original MSRP from the catalog instead of '
                         'the PPO pricing agent\'s optimized prices.')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=7860)
    return ap.parse_args()


ARGS = _parse_args()
USE_RERANKER = not ARGS.retriever_only
USE_OPTIMIZED_PRICES = not ARGS.static_prices

WORKSPACE = os.environ.get('PLAYBOOK_WORKSPACE', os.path.expanduser('~'))
DATA_DIR = os.path.join(WORKSPACE, "data")
MODELS_DIR = Path(WORKSPACE) / "models"

# Optional Nemotron client (gracefully degrades if server not running)
try:
    from nemotron_client import get_explanation_or_fallback
    HAS_NEMOTRON_CLIENT = True
except ImportError:
    HAS_NEMOTRON_CLIENT = False

# --- Load item metadata ---
print("Loading data...")
item_meta = pd.read_parquet(os.path.join(DATA_DIR, "processed", "dress_metadata.parquet"))
interactions = pd.read_parquet(os.path.join(DATA_DIR, "processed", "dress_interactions.parquet"))
print(f"  Raw: {len(item_meta):,} items, {len(interactions):,} interactions")

# Serve-time non-dress filter. The upstream `prepare_data.py` filter is a
# loose substring match that admits ~244 items obviously not dresses
# (socks, jackets, men's pants — see findings.md). Trust the upstream
# positive selection; just subtract obvious false-positives here.
_drop_mask = item_meta['title'].fillna('').apply(looks_non_dress)
_n_dropped = int(_drop_mask.sum())
item_meta = item_meta.loc[~_drop_mask].reset_index(drop=True)
_valid_ids = set(item_meta['item_id'])
interactions = interactions[interactions['item_id'].isin(_valid_ids)]
interactions = interactions.sort_values(['user_id', 'timestamp'])
print(f"  Filtered: dropped {_n_dropped:,} non-dress items; "
      f"{len(item_meta):,} items, {len(interactions):,} interactions remain")

# --- PPO-optimized prices (default) -----------------------------------------
# The pricing agent's trained PPO policy chooses a per-item price multiplier;
# we evaluate it once at day=0 on a fresh inventory state and cache the result
# so /api/recommend stays a cheap read. --static-prices skips this entirely
# and falls back to the catalog's MSRP.
optimized_prices: dict[str, float] = {}
if USE_OPTIMIZED_PRICES:
    from pricing_agent import (
        PRICING_CONFIG,
        InventoryState,
        PPOPolicy,
        _build_actor_critic,
        load_amazon_dresses_catalog,
        load_checkpoint,
    )

    PPO_CKPT = MODELS_DIR / "pricing_ppo" / "policy.pt"
    if not PPO_CKPT.exists():
        raise FileNotFoundError(
            f"PPO pricing checkpoint not found at {PPO_CKPT}.\n"
            "Train it first (`bash assets/pricing_agent.sh`) "
            "or launch with --static-prices to display the original MSRP."
        )

    print(f"Loading PPO pricing policy from {PPO_CKPT}...")
    import torch  # local; pricing_agent imports torch lazily
    _ppo_ckpt = load_checkpoint(PPO_CKPT)
    _multipliers = np.asarray(_ppo_ckpt["multipliers"], dtype=np.float64)
    _ppo_catalog = load_amazon_dresses_catalog(n_items=0, seed=0)
    _ppo_state = InventoryState.initialize(_ppo_catalog, PRICING_CONFIG, seed=0)
    _ppo_device = "cuda" if torch.cuda.is_available() else "cpu"
    _ppo_net = _build_actor_critic(len(_multipliers)).to(_ppo_device)
    _ppo_net.load_state_dict(_ppo_ckpt["state_dict"])
    _ppo_policy = PPOPolicy(
        net=_ppo_net,
        multipliers=_multipliers,
        device=_ppo_device,
        horizon=_ppo_ckpt["config"]["horizon"],
        price_norm=float(_ppo_state.base_prices.max()),
        inv_norm=float(_ppo_state._initial_inventories.max()),
        greedy=True,
    )
    _ppo_day0 = _ppo_policy.select_prices(_ppo_state, day=0)
    optimized_prices = {
        str(iid): float(p)
        for iid, p in zip(_ppo_catalog["item_id"].tolist(), _ppo_day0)
    }
    print(f"  Optimized prices computed for {len(optimized_prices):,} items")
else:
    print("Static prices mode (--static-prices): showing catalog MSRP.")

item_lookup = {}
for _, row in item_meta.iterrows():
    iid = row['item_id']
    base = float(row['price']) if pd.notna(row.get('price')) else None
    if USE_OPTIMIZED_PRICES and str(iid) in optimized_prices:
        display_price = optimized_prices[str(iid)]
    else:
        display_price = base
    item_lookup[iid] = {
        'title': str(row.get('title', ''))[:55],
        'price': display_price,
        'base_price': base,
        'image_url': str(row.get('image_url', '')),
    }

user_histories = {}
for uid, group in interactions.groupby('user_id'):
    user_histories[uid] = group.to_dict('records')

good_users_raw = sorted(
    [(uid, len(h)) for uid, h in user_histories.items() if len(h) >= 10],
    key=lambda x: -x[1]
)[:200]

# Each of the 200 displayed users gets a stable, UNIQUE women's name
# (no hash collisions). Assigned by alphabetical user_id so it's
# deterministic across restarts regardless of interaction-count ties.
NAME_POOL = [
    'Aaliyah','Abigail','Adelina','Aisha','Alaia','Alessia','Alina','Amara','Amaya','Amelia',
    'Amira','Anais','Andrea','Anika','Anya','Aoife','Aria','Ariana','Astrid','Aurora',
    'Ava','Aya','Ayla','Azure','Beatrix','Belen','Bianca','Briar','Brigitte','Calla',
    'Camila','Carmen','Cassia','Celeste','Charlotte','Chiara','Chloe','Claire','Clara','Constance',
    'Coralie','Dahlia','Daniela','Daphne','Delfina','Devi','Dunja','Eden','Edith','Elena',
    'Eliana','Elif','Elin','Elisa','Eloise','Elsa','Ember','Emilia','Emma','Esme',
    'Esra','Estelle','Eva','Evelyn','Farah','Fatima','Faye','Felicia','Fiona','Florence',
    'Freya','Frida','Gabriela','Genevieve','Giulia','Greta','Gwen','Hadley','Hana','Hannah',
    'Harlow','Hazel','Helena','Hira','Iara','Ida','Imani','Imogen','Inara','Ingrid',
    'Iris','Isabella','Isadora','Isla','Ivy','Iza','Jade','Jana','Jasmine','Josephine',
    'Juliette','June','Juno','Kaia','Kalia','Kara','Karina','Katia','Kavya','Khloe',
    'Kira','Kiri','Lailah','Lana','Larisa','Lavinia','Layla','Leila','Lena','Lila',
    'Liliana','Lily','Linnea','Lior','Liv','Liya','Lola','Lorena','Luna','Lyla',
    'Maeve','Magnolia','Maja','Malia','Manon','Mara','Marcela','Margot','Maria','Marina',
    'Marisol','Marlowe','Maya','Mei','Melisa','Mia','Mila','Mira','Miriam','Moana',
    'Nadia','Naia','Naomi','Nara','Natasha','Nia','Niamh','Nika','Nina','Nisha',
    'Nora','Noor','Nova','Oksana','Olivia','Ophelia','Paloma','Penelope','Petra','Phoebe',
    'Priya','Quinn','Quintessa','Rachel','Raina','Raisa','Reema','Renata','Rhea','Riya',
    'Romy','Rosalie','Ruby','Saanvi','Saba','Saoirse','Sasha','Selena','Selma','Senna',
    'Serena','Shira','Sienna','Simone','Sloan','Sofia','Soleil','Sophia','Stella','Talia',
]
assert len(NAME_POOL) == 200, f"NAME_POOL must have exactly 200 entries (has {len(NAME_POOL)})"
assert len(set(NAME_POOL)) == len(NAME_POOL), "NAME_POOL has duplicates"
_sorted_uids = sorted(uid for uid, _ in good_users_raw)
_uid_to_name = {uid: NAME_POOL[i] for i, uid in enumerate(_sorted_uids)}

good_users = [(uid, count, _uid_to_name[uid]) for uid, count in good_users_raw]

item_popularity = interactions['item_id'].value_counts().index.tolist()

# --- Load trained HLLM embeddings + FAISS index ---
print("Loading HLLM embeddings...")
hllm_embeddings = np.load(os.path.join(DATA_DIR, "processed", "hllm_item_embeddings.npy")).astype(np.float32)
hllm_id_map = np.load(os.path.join(DATA_DIR, "processed", "hllm_item_id_map.npy"), allow_pickle=True)

# Build item_id → HLLM index mapping (skip padding at index 0)
item_to_hllm_idx = {str(iid): i for i, iid in enumerate(hllm_id_map) if iid != '[PAD]'}
hllm_idx_to_item = {i: str(iid) for i, iid in enumerate(hllm_id_map) if iid != '[PAD]'}

# Build FAISS index over item embeddings (skip padding row 0)
# Use all rows including padding for index consistency (HLLM uses 1-indexed)
print("Building FAISS index...")
faiss_index = faiss.IndexFlatIP(hllm_embeddings.shape[1])
faiss_index.add(hllm_embeddings)
print(f"  FAISS index: {faiss_index.ntotal:,} vectors, {hllm_embeddings.shape[1]} dims")

# Build user embeddings from purchase history (mean of item embeddings)
print("Building user embeddings...")
user_embeddings = {}
for uid, records in user_histories.items():
    item_ids = [r['item_id'] for r in records]
    idxs = [item_to_hllm_idx[iid] for iid in item_ids if iid in item_to_hllm_idx]
    if idxs:
        emb = hllm_embeddings[idxs].mean(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        user_embeddings[uid] = emb
print(f"  {len(user_embeddings):,} user embeddings built")

# --- Re-ranker (default: LightGBM; bypass with --retriever-only) ---
RERANKER = None
ITEM_FEAT_ARR = None
USER_SCALARS = {}
RETRIEVAL_TOP_K = 100  # candidates pulled from FAISS before rerank
FINAL_TOP_K = 4        # items returned to the UI (top-4 gives recs more room beside the sidebar)

if USE_RERANKER:
    from train_reranker_lightgbm import (
        FEATURE_COLS, build_user_samples, compute_item_stats, compute_user_stats,
    )

    LGBM_PATH = MODELS_DIR / "reranker_lightgbm" / "reranker_lightgbm.txt"
    if not LGBM_PATH.exists():
        raise FileNotFoundError(
            f"LightGBM re-ranker checkpoint not found at {LGBM_PATH}.\n"
            "Train it first (`bash assets/train_reranker.sh`) "
            "or launch with --retriever-only to skip the re-ranker."
        )
    print(f"Loading LightGBM re-ranker from {LGBM_PATH}...")
    RERANKER = lgb.Booster(model_file=str(LGBM_PATH))

    print("Precomputing item / user feature tables for re-ranker...")
    _t = time.time()
    _item_stats = compute_item_stats(interactions, item_meta)
    _item_idx_to_id = [str(i) for i in hllm_id_map]
    _item_feat_cols = [
        'item_total_purchases', 'item_unique_buyers',
        'item_pop_30d', 'item_pop_90d', 'item_pop_180d', 'item_trend',
        'item_recency_days', 'item_age_days',
        'log_price', 'title_length', 'desc_length', 'has_image',
    ]
    ITEM_FEAT_ARR = (
        _item_stats.reindex(_item_idx_to_id).fillna(0)[_item_feat_cols]
        .to_numpy(dtype=np.float32)
    )

    # Build user samples in the format compute_user_stats expects. For inference
    # we use the FULL history (no leave-last-out), so hist_idxs == idxs.
    _item_to_idx = {str(iid): i for i, iid in enumerate(hllm_id_map) if iid != '[PAD]'}
    _samples_for_inference = []
    for _uid, _records in user_histories.items():
        _seq = [r['item_id'] for r in _records]
        _ts = [int(r['timestamp']) for r in _records]
        _idxs = [_item_to_idx[i] for i in _seq if i in _item_to_idx]
        _ts_kept = [t for i, t in zip(_seq, _ts) if i in _item_to_idx]
        if _idxs:
            _samples_for_inference.append((_uid, _idxs, _idxs[-1], _ts_kept))
    USER_SCALARS = compute_user_stats(_samples_for_inference, item_meta, _item_idx_to_id)
    _LOG_PRICE_COL = _item_feat_cols.index('log_price')

    _USER_HISTORY_HLLM_IDX = {
        uid: _idxs for (uid, _idxs, _, _) in _samples_for_inference
    }
    print(f"  Precompute done in {time.time()-_t:.1f}s "
          f"({RERANKER.num_trees()} trees, {len(FEATURE_COLS)} features, "
          f"{len(USER_SCALARS):,} user scalar rows)")
else:
    print("Re-ranker disabled (--retriever-only).")


def _rerank_candidates(
    user_id: str,
    cand_hllm_idxs: np.ndarray,
    cand_scores: np.ndarray,
) -> np.ndarray:
    """Score candidates with LightGBM. Returns indices into ``cand_hllm_idxs``
    sorted by descending model score. Caller filters / takes top-K.

    Vectorized over candidates only (one user per request, K~=100).
    """
    history = _USER_HISTORY_HLLM_IDX.get(user_id, [])
    K = len(cand_hllm_idxs)
    F = len(FEATURE_COLS)
    f2p = {c: i for i, c in enumerate(FEATURE_COLS)}
    X = np.zeros((K, F), dtype=np.float32)

    # HLLM signals
    X[:, f2p['hllm_dot_product']] = cand_scores
    if history:
        recent = history[-10:]
        hist_emb = hllm_embeddings[recent]                        # (h, dim)
        cand_emb = hllm_embeddings[cand_hllm_idxs]                # (K, dim)
        sims = cand_emb @ hist_emb.T                              # (K, h)
        X[:, f2p['hllm_max_hist_sim']] = sims.max(axis=1)
        X[:, f2p['hllm_avg_hist_sim']] = sims.mean(axis=1)

    # Item-side
    item_block = ITEM_FEAT_ARR[cand_hllm_idxs]                    # (K, 12)
    for c, name in enumerate(_item_feat_cols):
        X[:, f2p[name]] = item_block[:, c]

    # User-side
    us = USER_SCALARS.get(user_id, {})
    for col in ('user_total_purchases', 'user_unique_items',
                'user_avg_price', 'user_price_std', 'user_recency_days'):
        X[:, f2p[col]] = us.get(col, 0.0)

    # Cross
    cand_price = np.expm1(item_block[:, _LOG_PRICE_COL])
    u_avg = us.get('user_avg_price', 0.0)
    X[:, f2p['price_ratio']] = cand_price / (u_avg + 1e-8)
    X[:, f2p['price_diff']] = cand_price - u_avg

    hist_set = set(history)
    X[:, f2p['is_repurchase']] = np.fromiter(
        (1.0 if int(c) in hist_set else 0.0 for c in cand_hllm_idxs),
        dtype=np.float32, count=K,
    )

    scores = RERANKER.predict(X)
    return np.argsort(-scores), scores


app = FastAPI()


_gpu_cache = {'data': None, 'ts': 0}

def gpu_stats():
    """GPU stats with 5-second cache to avoid nvidia-smi overhead on every request."""
    now = time.time()
    if _gpu_cache['data'] and now - _gpu_cache['ts'] < 5:
        return _gpu_cache['data']
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits', '-i=1'],
            capture_output=True, text=True, timeout=5)
        p = r.stdout.strip().split(', ')
        data = {'util': float(p[0]), 'vram_used': float(p[1])/1024,
                'vram_total': float(p[2])/1024, 'temp': float(p[3]), 'power': float(p[4])}
    except:
        data = {'util': 0, 'vram_used': 0, 'vram_total': 252, 'temp': 0, 'power': 0}
    _gpu_cache['data'] = data
    _gpu_cache['ts'] = now
    return data


@app.get("/api/users")
def api_users():
    return [{"id": uid, "count": c, "name": name} for uid, c, name in good_users]


@app.get("/api/history/{user_id}")
def api_history(user_id: str):
    history = user_histories.get(user_id, [])[-5:]
    return [{
        'item_id': h['item_id'],
        'title': item_lookup.get(h['item_id'], {}).get('title', ''),
        'price': item_lookup.get(h['item_id'], {}).get('price'),
        'base_price': item_lookup.get(h['item_id'], {}).get('base_price'),
        'image_url': item_lookup.get(h['item_id'], {}).get('image_url', ''),
        'date': datetime.fromtimestamp(int(h['timestamp'])).strftime('%b %Y'),
    } for h in history]


@app.get("/api/recommend/{user_id}")
def api_recommend(user_id: str):
    t0 = time.time()
    history = user_histories.get(user_id, [])
    user_items = set(h['item_id'] for h in history)

    lat_retriever_ms = 0.0
    lat_reranker_ms = 0.0
    lat_explainer_ms = 0.0

    user_emb = user_embeddings.get(user_id)
    if user_emb is not None:
        t = time.time()
        query = user_emb.reshape(1, -1).astype(np.float32)
        distances, indices = faiss_index.search(query, RETRIEVAL_TOP_K)
        lat_retriever_ms = (time.time() - t) * 1000
        cand_hllm_idxs = indices[0]
        cand_scores = distances[0].astype(np.float32)

        if USE_RERANKER:
            t = time.time()
            order, rerank_scores = _rerank_candidates(user_id, cand_hllm_idxs, cand_scores)
            lat_reranker_ms = (time.time() - t) * 1000
            display_scores = rerank_scores
            method = "HLLM + LightGBM"
        else:
            order = np.arange(len(cand_hllm_idxs))
            display_scores = cand_scores
            method = "HLLM"

        candidates = []
        scores = []
        for rank in order:
            idx = int(cand_hllm_idxs[rank])
            item_id = hllm_idx_to_item.get(idx)
            # Require item to be in the filtered metadata (drops non-dress
            # items that the unfiltered FAISS index still surfaces) and not
            # already in the user's history.
            if item_id and item_id in item_lookup and item_id not in user_items:
                candidates.append(item_id)
                scores.append(float(display_scores[rank]))
            if len(candidates) >= FINAL_TOP_K:
                break
    else:
        # Cold-start fallback: popularity (re-ranker requires a user embedding)
        candidates = [i for i in item_popularity if i not in user_items][:FINAL_TOP_K]
        scores = [0.97 - i * 0.04 for i in range(len(candidates))]
        method = "Popularity"

    recs = []
    for item_id, score in zip(candidates, scores):
        info = item_lookup.get(item_id, {})
        recs.append({
            'item_id': item_id,
            'title': info.get('title', ''),
            'price': info.get('price'),
            'base_price': info.get('base_price'),
            'image_url': info.get('image_url', ''),
            'score': round(score, 3),
        })

    # Build style-focused fallback (used only if Nemotron unreachable)
    recent_titles = [item_lookup.get(h['item_id'], {}).get('title', '') for h in history[-3:]]
    recent_titles = [t for t in recent_titles if t]
    if recent_titles:
        fallback = (
            "These dresses share style characteristics with your recent picks — "
            "similar cuts, silhouettes, and visual themes identified by the HLLM retriever "
            "and ranked by the LightGBM re-ranker."
        )
    else:
        fallback = "Showing popular dresses. Purchase a few items for personalized style matches."

    # Try Nemotron explanation (graceful fallback if Ollama unreachable)
    llm_generated = False
    if HAS_NEMOTRON_CLIENT and recent_titles:
        hist_for_llm = [{'title': item_lookup.get(h['item_id'], {}).get('title', ''),
                         'price': item_lookup.get(h['item_id'], {}).get('price')}
                        for h in history[-5:]]
        t = time.time()
        explanation, llm_generated = get_explanation_or_fallback(hist_for_llm, recs, fallback)
        lat_explainer_ms = (time.time() - t) * 1000
    else:
        explanation = fallback

    gpu = gpu_stats()
    return {
        'recommendations': recs,
        'explanation': explanation,
        'llm_generated': llm_generated,
        'method': method,
        'latency_ms': round((time.time() - t0) * 1000),
        'latency_retriever_ms': round(lat_retriever_ms, 1),
        'latency_reranker_ms': round(lat_reranker_ms, 1),
        'latency_explainer_ms': round(lat_explainer_ms),
        'reranker_active': USE_RERANKER,
        'gpu': gpu,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NVIDIA Fashion · Personalized Recommendations</title>
<style>
  :root {
    --bg: #fafafa;
    --surface: #ffffff;
    --surface-2: #f3f3f3;
    --border: #e5e5e5;
    --border-strong: #d0d0d0;
    --text: #111111;
    --text-muted: #5f5f5f;
    --text-dim: #9a9a9a;
    --accent: #76b900;
    --accent-hover: #6aa600;
    --star: #f5a623;

    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --mono: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Liberation Mono", monospace;

    --t-micro: 11px;
    --t-meta:  12px;
    --t-body:  13px;
    --t-title: 15px;
    --t-heading: 18px;
    --t-hero: 24px;

    --s-1: 4px;
    --s-2: 8px;
    --s-3: 12px;
    --s-4: 16px;
    --s-5: 20px;
    --s-6: 32px;
    --s-7: 48px;

    --radius: 6px;
    --radius-lg: 10px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }
  html { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
  body {
    font-family: var(--font);
    font-size: var(--t-body);
    color: var(--text);
    background: var(--bg);
    line-height: 1.5;
  }
  a { color: inherit; text-decoration: none; }

  /* ─── Utility bar ─────────────────────────────────────── */
  .utility-bar {
    background: #111;
    color: #f5f5f5;
    font-size: var(--t-micro);
    padding: 6px var(--s-6);
    display: flex; justify-content: space-between; align-items: center;
  }
  .utility-bar .util-center { flex: 1; text-align: center; letter-spacing: 0.04em; }
  .utility-bar a { color: #d4d4d4; }
  .utility-bar a:hover { color: #fff; }

  /* ─── Navbar ──────────────────────────────────────────── */
  .navbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: var(--s-3) var(--s-6);
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: var(--s-6);
  }
  .navbar .brand {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: var(--t-heading);
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  .navbar .brand .dot { color: var(--accent); font-size: 12px; line-height: 0; }
  .navbar .brand .word { color: var(--text); }
  .navbar .brand .subword { color: var(--text-muted); font-weight: 500; }
  .navbar .cat-nav {
    display: flex; gap: var(--s-5);
    font-size: var(--t-body);
    color: var(--text-muted);
  }
  .navbar .cat-nav a:hover { color: var(--text); }
  .navbar .nav-right {
    display: flex; align-items: center; gap: var(--s-4);
  }
  .navbar .searchbox {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 6px 14px;
    width: 320px;
    font-size: var(--t-meta);
    color: var(--text);
    font-family: inherit;
  }
  .navbar .searchbox::placeholder { color: var(--text-dim); }
  .navbar .icon-link {
    display: inline-flex; flex-direction: column; align-items: center;
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.1;
  }
  .navbar .icon-link .glyph { font-size: 18px; }

  /* ─── Hero ────────────────────────────────────────────── */
  .container {
    max-width: 1440px;
    margin: 0 auto;
    padding: var(--s-3) var(--s-6) var(--s-3);
  }
  .hero {
    display: flex; align-items: center; justify-content: space-between;
    gap: var(--s-5);
    margin-bottom: var(--s-3);
  }
  .hero h1 {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.015em;
  }
  .hero h1 .edit { color: var(--text-dim); margin-left: 6px; font-size: 14px; cursor: pointer; }
  .hero-actions { display: flex; gap: var(--s-3); align-items: center; }
  .hero-actions select {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    padding: 9px 12px;
    border-radius: var(--radius);
    font-size: var(--t-meta);
    color: var(--text);
    font-family: inherit;
    min-width: 240px;
  }
  .hero-actions button {
    padding: 9px 18px;
    background: #111;
    color: #fff;
    border: none;
    border-radius: var(--radius);
    font-size: var(--t-meta);
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
  }
  .hero-actions button:hover { background: #2a2a2a; }
  .hero-actions button:disabled { background: var(--text-dim); cursor: wait; }

  /* ─── Section ─────────────────────────────────────────── */
  .section { margin-bottom: var(--s-3); }
  .section-head {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: var(--s-2);
  }
  .section-head h2 {
    font-size: var(--t-heading);
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  .section-head .view-all {
    font-size: var(--t-meta);
    color: var(--text-muted);
    text-decoration: underline;
  }

  /* ─── Product grid ────────────────────────────────────── */
  .product-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: var(--s-4);
  }
  .product {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    transition: box-shadow .15s ease, border-color .15s ease;
    position: relative;
  }
  .product:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.06); border-color: var(--border-strong); }
  .product .img {
    width: 100%;
    aspect-ratio: 1/1;
    background: #fff;
    overflow: hidden;
    position: relative;
  }
  /* Recommendation cards inherit 1:1 from .product .img above. With top-4
     recs the cards are wide enough (~270px) that 1:1 + object-fit: contain
     shows the full dress without truncation. */
  /* Show the whole product photo — no top/bottom cropping. Letterbox sides
     against the card background if the image's natural aspect differs. */
  .product .img img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .no-img {
    height: 100%;
    display: flex; align-items: center; justify-content: center;
    color: var(--text-dim);
    font-size: var(--t-micro);
  }
  .product .heart {
    position: absolute;
    top: 10px; right: 10px;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 50%;
    width: 30px; height: 30px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    transition: background .15s, color .15s;
    font-size: 15px;
    color: #555;
  }
  .product .heart:hover { background: rgba(255,255,255,0.95); color: #d62828; }
  .product .info { padding: var(--s-2) var(--s-3) var(--s-3); }
  .product .title {
    font-size: var(--t-meta);
    color: var(--text);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: var(--s-1);
    line-height: 1.35;
    min-height: calc(var(--t-meta) * 1.35 * 2);
  }
  .product .price {
    font-size: var(--t-title);
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
  }
  .product .price .delta {
    font-size: var(--t-micro);
    font-weight: 600;
    letter-spacing: 0.01em;
  }
  .product .price .delta.up   { color: #1b8f3a; }
  .product .price .delta.down { color: #c0392b; }
  .product .rating {
    margin-top: var(--s-1);
    display: flex; align-items: center; gap: 4px;
    font-size: var(--t-micro);
    color: var(--text-muted);
  }
  .product .rating .stars { color: var(--star); letter-spacing: 1px; }
  .product .rating .count { color: var(--text-dim); }

  /* ─── Recommendations row: why-panel + 4 cards as a 5-column grid where
     all cells are the same width. #recs is layout-invisible so its 4 child
     product cards flow into columns 2-5 as siblings to the why-panel. */
  .recs-layout {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: var(--s-4);
    align-items: stretch;
  }
  .recs-layout > #recs { display: contents; }
  .why-panel {
    background: #f8f6f2;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--s-4);
    display: flex;
    flex-direction: column;
  }
  .why-panel ul { flex: 1; }   /* bullets fill the middle; link sits at bottom */
  .why-panel h3 {
    font-size: var(--t-body);
    font-weight: 700;
    margin-bottom: var(--s-3);
    display: inline-flex; align-items: center; gap: 6px;
  }
  .why-panel h3 .icon { color: var(--accent); }
  .why-panel ul {
    list-style: none;
    padding: 0;
    margin-bottom: var(--s-3);
  }
  .why-panel li {
    font-size: var(--t-meta);
    color: var(--text-muted);
    padding: 4px 0 4px 16px;
    position: relative;
    line-height: 1.5;
    text-align: justify;
    text-justify: inter-word;
    hyphens: auto;
  }
  .why-panel li::before {
    content: '';
    position: absolute;
    left: 0; top: 11px;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
  }
  .why-panel .see-how {
    font-size: var(--t-micro);
    color: var(--text-muted);
    text-decoration: underline;
    cursor: pointer;
  }
  .why-panel .source-line {
    font-size: var(--t-micro);
    color: var(--text-dim);
    margin-top: var(--s-3);
    padding-top: var(--s-3);
    border-top: 1px solid var(--border);
    display: none;
  }
  .why-panel .source-line.visible { display: block; }
  .why-panel .source-line b { color: var(--accent); }
  .why-panel .source-line .badge {
    display: inline-block;
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(118,185,0,0.12);
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    margin-left: 4px;
  }
  .why-panel .source-line .badge.fallback { background: var(--surface-2); color: var(--text-dim); border: 1px solid var(--border); }

  /* ─── Trust footer ────────────────────────────────────── */
  .trust-bar {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: var(--s-2) var(--s-6);
  }
  .trust-bar-inner {
    max-width: 1440px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--s-5);
  }
  .trust-cell {
    display: flex; align-items: center; gap: var(--s-3);
  }
  .trust-cell .glyph {
    width: 34px; height: 34px;
    display: inline-flex; align-items: center; justify-content: center;
    background: var(--surface-2);
    border-radius: 50%;
    color: var(--text);
    flex: 0 0 34px;
  }
  .trust-cell .label {
    font-size: var(--t-meta);
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
  }
  .trust-cell .sub {
    font-size: var(--t-micro);
    color: var(--text-muted);
  }

  /* ─── System info strip (tiny) ───────────────────────── */
  .sysinfo {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 8px var(--s-6);
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    display: flex; flex-wrap: wrap; gap: var(--s-4); align-items: center;
  }
  .sysinfo .lbl { color: var(--text-dim); }
  .sysinfo .val { color: var(--text-muted); font-weight: 500; }
  .sysinfo .sep { color: var(--border-strong); }

  /* ─── Responsive: switch from grid to horizontal scroll at narrow widths
     so cards stay full-size and the user swipes/scrolls instead of wrapping. */
  @media (max-width: 1280px) {
    .product-grid {
      display: flex;
      grid-template-columns: none;
      overflow-x: auto;
      scroll-snap-type: x mandatory;
      gap: var(--s-3);
      padding-bottom: var(--s-2);
      -webkit-overflow-scrolling: touch;
    }
    .product-grid::-webkit-scrollbar { height: 6px; }
    .product-grid::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
    .product-grid .product {
      flex: 0 0 200px;
      scroll-snap-align: start;
    }
    /* In horizontal-scroll mode, also collapse the recs-layout grid and
       reuse the same overflow strategy. */
    .recs-layout {
      display: flex;
      grid-template-columns: none;
      overflow-x: auto;
      scroll-snap-type: x mandatory;
      gap: var(--s-3);
      padding-bottom: var(--s-2);
    }
    .recs-layout > .why-panel { flex: 0 0 200px; scroll-snap-align: start; }
    .recs-layout > #recs { display: contents; }
    .recs-layout > #recs > .product { flex: 0 0 200px; scroll-snap-align: start; }
  }
  @media (max-width: 1024px) {
    .navbar { grid-template-columns: auto 1fr auto; gap: var(--s-4); padding: var(--s-3) var(--s-4); }
    .navbar .cat-nav { display: none; }
    .navbar .searchbox { width: 200px; }
    .container { padding: var(--s-4) var(--s-4); }
    .trust-bar-inner { grid-template-columns: repeat(2, 1fr); gap: var(--s-3); }
  }
  @media (max-width: 640px) {
    .navbar { grid-template-columns: auto auto; }
    .navbar .nav-right .icon-link { display: none; }
    .navbar .searchbox { display: none; }
    .hero { flex-direction: column; align-items: stretch; }
    .hero-actions { flex-wrap: wrap; }
    .hero-actions select { min-width: 0; flex: 1; }
    .utility-bar .util-right { display: none; }
  }
</style>
</head>
<body>

<div class="utility-bar">
  <span class="util-left">●</span>
  <span class="util-center">Free shipping on orders over $75</span>
  <a href="#" class="util-right">Download our app for exclusive offers ↗</a>
</div>

<nav class="navbar">
  <a href="#" class="brand">
    <span class="dot">●</span>
    <span class="word">NVIDIA</span>
    <span class="subword">Fashion</span>
  </a>
  <div class="cat-nav">
    <a href="#">Women's</a>
    <a href="#">Men's</a>
    <a href="#">Kids</a>
    <a href="#">Sale</a>
    <a href="#">Inspiration</a>
  </div>
  <div class="nav-right">
    <input class="searchbox" type="search" placeholder="Search dresses, brands, more…" aria-label="Search">
    <a href="#" class="icon-link" title="Account"><span class="glyph">👤</span>Account</a>
    <a href="#" class="icon-link" title="Wishlist"><span class="glyph">♡</span>Wishlist</a>
    <a href="#" class="icon-link" title="Cart"><span class="glyph">🛍</span>Cart</a>
  </div>
</nav>

<main class="container">
  <div class="hero">
    <h1>Welcome back, <span id="greet-name">there</span><span class="edit" title="Personalized picks">✨</span></h1>
    <div class="hero-actions">
      <select id="user-select" aria-label="User"></select>
      <button id="rec-btn" onclick="recommend()">Refresh picks</button>
    </div>
  </div>

  <section class="section">
    <div class="section-head">
      <h2>Your recent purchases</h2>
      <a href="#" class="view-all">View all →</a>
    </div>
    <div id="history" class="product-grid"></div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Recommended for you</h2>
    </div>
    <div class="recs-layout">
      <aside class="why-panel">
        <h3><span class="icon">✦</span>Why these picks?</h3>
        <ul id="why-list">
          <li>Loading style match…</li>
        </ul>
        <a class="see-how" onclick="document.getElementById('source-line').classList.toggle('visible'); return false;">See how it works</a>
        <div class="source-line" id="source-line">
          <span id="method-label">HLLM + LightGBM</span> retriever &amp; re-ranker →
          <b>Nemotron Mini</b> explainer
          <span class="badge" id="llm-badge">live</span>
        </div>
      </aside>
      <div id="recs" class="product-grid"></div>
    </div>
  </section>
</main>

<footer class="trust-bar">
  <div class="trust-bar-inner">
    <div class="trust-cell">
      <span class="glyph"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="1.5" y="6" width="13" height="10" rx="1"/><path d="M14.5 9h3.5l3 3v4h-6.5z"/><circle cx="5.5" cy="17" r="2"/><circle cx="17.5" cy="17" r="2"/></svg></span>
      <div><div class="label">Free delivery</div><div class="sub">On orders over $75</div></div>
    </div>
    <div class="trust-cell">
      <span class="glyph"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 16.5-5"/><path d="M21 4v5h-5"/><path d="M21 12a9 9 0 0 1-16.5 5"/><path d="M3 20v-5h5"/></svg></span>
      <div><div class="label">Easy returns</div><div class="sub">30-day window</div></div>
    </div>
    <div class="trust-cell">
      <span class="glyph"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="1.5"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg></span>
      <div><div class="label">Secure checkout</div><div class="sub">256-bit SSL</div></div>
    </div>
    <div class="trust-cell">
      <span class="glyph"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12z"/></svg></span>
      <div><div class="label">Customer support</div><div class="sub">7 days a week</div></div>
    </div>
  </div>
</footer>

<div class="sysinfo">
  <span class="lbl">● NVIDIA DGX Station GB300</span>
  <span class="sep">·</span>
  <span><span class="lbl">retr</span> <span class="val" id="lat-retr">—</span></span>
  <span><span class="lbl">rerank</span> <span class="val" id="lat-rerank">—</span></span>
  <span><span class="lbl">llm</span> <span class="val" id="lat-llm">—</span></span>
  <span class="sep">·</span>
  <span><span class="lbl">gpu</span> <span class="val" id="hw-util">—</span></span>
  <span><span class="lbl">vram</span> <span class="val" id="hw-vram">—</span></span>
  <span><span class="lbl">temp</span> <span class="val" id="hw-temp">—</span></span>
  <span><span class="lbl">power</span> <span class="val" id="hw-power">—</span></span>
</div>

<script>
const $ = id => document.getElementById(id);

// Server assigns unique women's names per user (one of 200, no collisions).
// Loaded via /api/users into USER_NAME_MAP.
const USER_NAME_MAP = {};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Stable 32-bit-ish hash for deterministic synthesis
function strHash(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function nameFor(uid) { return USER_NAME_MAP[uid] || 'there'; }

function ratingFor(itemId) {
  const h = strHash(itemId || 'x');
  const r = 4.0 + (h % 10) / 10;             // 4.0 .. 4.9
  const reviews = 80 + (h % 9000);           // 80 .. 9079
  return { value: r, reviews };
}

function starGlyphs(r) {
  const full = Math.floor(r);
  const half = (r - full) >= 0.5;
  const empty = 5 - full - (half ? 1 : 0);
  return '★'.repeat(full) + (half ? '½' : '') + '☆'.repeat(empty);
}

function fmtLatency(ms) {
  if (ms == null || ms === 0) return '—';
  const n = Number(ms);
  if (n < 1)   return n.toFixed(2) + 'ms';
  if (n < 100) return n.toFixed(1) + 'ms';
  return Math.round(n) + 'ms';
}

function priceDeltaHTML(base, current) {
  if (base == null || current == null || base <= 0) return '';
  const pct = ((current - base) / base) * 100;
  if (Math.abs(pct) < 0.05) return '';
  const up = pct > 0;
  const arrow = up ? '↑' : '↓';
  const cls = up ? 'up' : 'down';
  return `<span class="delta ${cls}" title="vs. MSRP $${Number(base).toFixed(2)}">${arrow} ${Math.abs(pct).toFixed(1)}%</span>`;
}

function cardHTML(item) {
  const rating = ratingFor(item.item_id || item.title || '');
  const priceText = item.price != null ? '$' + Number(item.price).toFixed(2) : '—';
  const delta = priceDeltaHTML(item.base_price, item.price);
  return `<article class="product">
    <div class="img">
      ${item.image_url
        ? `<img src="${escapeHtml(item.image_url)}" alt="" loading="lazy" onerror="this.parentElement.innerHTML+='<div class=no-img>No image</div>'">`
        : '<div class="no-img">No image</div>'}
      <span class="heart" title="Save to wishlist">♡</span>
    </div>
    <div class="info">
      <div class="title">${escapeHtml(item.title || '')}</div>
      <div class="price"><span>${priceText}</span>${delta}</div>
      <div class="rating">
        <span class="stars">${starGlyphs(rating.value)}</span>
        <span>${rating.value.toFixed(1)}</span>
        <span class="count">(${rating.reviews.toLocaleString()})</span>
      </div>
    </div>
  </article>`;
}

function parseBullets(text) {
  // Nemotron returns 3 phrases tagged [1] [2] [3]. Robust to multiple shapes.
  const s = String(text || '').trim();
  let parts;
  if (/\[[1-9]\]/.test(s)) {
    parts = s.split(/\s*\[[1-9]\]\s*/).filter(p => p.length > 0);
  } else if (s.includes('\n')) {
    parts = s.split(/\r?\n/);
  } else {
    parts = s.split(/[,;]\s+/);
  }
  return parts
    .map(p => p.replace(/^[-•*\d+.)\s]+/, '').trim())
    .map(cleanBullet)
    .filter(p => p.length > 0)
    .slice(0, 4);
}

// Strip residual model artifacts — small 4B model drifts onto specific
// phrasings ("in customer history and recommendations", "across both
// collections") despite explicit prohibition. Defensive normalization.
function cleanBullet(s) {
  // List of meta-reference phrases the model repeatedly emits despite
  // the prompt forbidding them. Strip them all (case-insensitive).
  const META_PHRASES = [
    /\s+in (customer|user)\s+history\s+and\s+recommendations/gi,
    /\s+in (customer|user)\s+history/gi,
    /\s+and\s+recommendations/gi,
    /\s+across\s+both\s+(collections|histories|sets|items?)/gi,
    /\s+across\s+all\s+items?\s+listed\s+above/gi,
    /\s+in\s+both\s+(collections|histories|sets)/gi,
    /\s+on\s+dresses\s+like\s+[^.]+/gi,
    /\s+naturally\s+extending\s+to\s+[^,.]+/gi,
    /\s+as\s+seen\s+in\s+[^.]+/gi,
    /\s+making\s+(it|them)\s+(harmonious|perfect)[^.]*/gi,
    /\s+from\s+the\s+same\s+season/gi,
    // "with LILLUSORY Womens Summer..." — strip when followed by an
    // ALL-CAPS brand name (drops everything until next period/end).
    /\s+with\s+[A-Z][A-Z]{2,}[^.]*/g,
    // "(e.g., something or another)" parenthetical hedges
    /\s*\(e\.g\.[^)]*\)/gi,
    // "but the style varies by customer preference" filler
    /\s*,?\s+but the style varies[^.]*/gi,
  ];
  let out = String(s);
  for (const re of META_PHRASES) out = out.replace(re, '');
  // Normalize ASCII arrow variants to the Unicode arrow we use in the prompt.
  out = out.replace(/\s*(-+>|=+>|—>|–>)\s*/g, ' → ');
  return out
    .replace(/\[([^\[\]]+)\]/g, '$1')         // [Foo Bar] -> Foo Bar
    .replace(/^[\]\)\d+.\-•*\s]+/, '')        // residual "] " / "1) " / "• " prefixes
    .replace(/\s*\([^)]*\b(verb|placeholder|trait|category)\b[^)]*\)\s*/gi, ' ')  // (pairing verb)
    .replace(/^([A-Z][^:]{3,80}):\s+/, '')    // strip "Item Name: " prefix
    .replace(/[:;,]\s*$/, '')                  // trailing punctuation
    .replace(/\s+/g, ' ')                      // collapse whitespace
    .trim();
}

async function loadUsers() {
  const users = await (await fetch('/api/users')).json();
  const sel = $('user-select');
  users.forEach(u => {
    USER_NAME_MAP[u.id] = u.name;
    const opt = document.createElement('option');
    opt.value = u.id;
    opt.textContent = u.name + ' · ' + u.count + ' purchases';
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => { updateGreeting(); loadHistory(); recommend(); });
  if (users.length) { updateGreeting(); loadHistory(); recommend(); }
}

function updateGreeting() {
  const uid = $('user-select').value;
  $('greet-name').textContent = nameFor(uid);
}

async function loadHistory() {
  const items = await (await fetch('/api/history/' + $('user-select').value)).json();
  $('history').innerHTML = items.map(cardHTML).join('');
}

async function recommend() {
  const btn = $('rec-btn');
  btn.disabled = true; btn.textContent = 'Loading…';
  const data = await (await fetch('/api/recommend/' + $('user-select').value)).json();

  $('recs').innerHTML = data.recommendations.map(cardHTML).join('');
  $('method-label').textContent = data.method;

  const bullets = parseBullets(data.explanation);
  if (bullets.length) {
    $('why-list').innerHTML = bullets.map(b => `<li>${escapeHtml(b)}</li>`).join('');
  } else {
    $('why-list').innerHTML = `<li>${escapeHtml(data.explanation || 'Personalized recommendations')}</li>`;
  }

  const badge = $('llm-badge');
  if (data.llm_generated) { badge.textContent = 'live'; badge.classList.remove('fallback'); }
  else                    { badge.textContent = 'fallback'; badge.classList.add('fallback'); }

  $('lat-retr').textContent   = fmtLatency(data.latency_retriever_ms);
  $('lat-rerank').textContent = data.reranker_active ? fmtLatency(data.latency_reranker_ms) : 'off';
  $('lat-llm').textContent    = fmtLatency(data.latency_explainer_ms);

  const g = data.gpu;
  $('hw-util').textContent  = g.util.toFixed(0) + '%';
  $('hw-vram').textContent  = g.vram_used.toFixed(0) + '/' + g.vram_total.toFixed(0) + 'GB';
  $('hw-temp').textContent  = g.temp.toFixed(0) + '°C';
  $('hw-power').textContent = g.power.toFixed(0) + 'W';

  btn.disabled = false; btn.textContent = 'Refresh picks';
}

loadUsers();
</script>
</body>
</html>"""

if __name__ == "__main__":
    retrieval_mode = "HLLM + LightGBM rerank" if USE_RERANKER else "HLLM retrieval only"
    pricing_mode = "PPO-optimized prices" if USE_OPTIMIZED_PRICES else "static MSRP"
    print(f"\nEnterprise Fashion Recommender ({retrieval_mode} · {pricing_mode})")
    print(f"   http://localhost:{ARGS.port}\n")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
