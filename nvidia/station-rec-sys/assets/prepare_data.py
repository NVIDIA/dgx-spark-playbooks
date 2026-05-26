"""Process Amazon Clothing → Dresses subset for HLLM training.

Filters to dress items only, then applies 5-core filtering.
Target: ~500K-800K interactions for 6-hour training on single GB300.

# [arXiv:2409.12740] HLLM data format: interactions CSV + item text CSV
"""

import json
import os
import re
import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# A loose `'dress' in title` substring filter admits "dress shoes", "dress
# pants", "dressy belts", etc. This regex pair was chosen by inspecting the
# 244 false positives the loose filter let through (see findings.md
# 2026-05-10): require a word-boundary match against dress-family terms AND
# reject items whose titles also match non-dress garment keywords.
_DRESS_RE = re.compile(r'\b(dress(es)?|sundress(es)?|gowns?)\b', re.IGNORECASE)
# Words below are dropped only when no dress-family word appears in the title
# (see looks_non_dress). Words like "shirt", "tank", "short", "cap", "blouse"
# are intentionally EXCLUDED because they appear inside legit dress titles
# ("T-shirt dress", "Tank dress", "Short sleeve dress", "Cap sleeve dress").
_NON_DRESS_RE = re.compile(
    r'\b('
    r'shoes?|boots?|sandals?|heels?|sneakers?|loafers?|'
    r'socks?|tights?|leggings?|'
    r'handbags?|purses?|totes?|clutches?|wallets?|'
    r'necklaces?|bracelets?|earrings?|watches?|'
    r'pants?|jeans?|shorts?|skirts?|'
    r'shirts?|blouses?|tunics?|tops|tees?|t-shirts?|'
    r'sweaters?|cardigans?|hoodies?|'
    r'jackets?|coats?|blazers?|vests?|'
    r'jumpsuits?|rompers?|'
    r'swimsuits?|bikinis?|'
    r'pajamas?|nightgowns?|kimonos?|sleepwear|underwear|panties'
    r')\b',
    re.IGNORECASE,
)


def is_dress(title: str, categories: str = '') -> bool:
    """Strict dress filter — requires a dress-family word AND no obvious
    non-dress garment word.

    Use at data-prep time to build a clean dataset from scratch. At serve
    time after `prepare_data.py`'s loose filter has already run, use
    `looks_non_dress` instead — strict-positive matching is too aggressive
    against legit dresses whose titles say only "Summer Maxi Long Sleeve".
    """
    text = f"{title} {categories}"
    if _NON_DRESS_RE.search(text):
        return False
    return bool(_DRESS_RE.search(text))


def looks_non_dress(title: str) -> bool:
    """Negative-only check for serve-time filtering.

    Returns True if the title's *head noun* is a non-dress garment.
    Uses an "appears last wins" heuristic: in "T-Shirt Dress" the head is
    Dress (keep), in "Mary Jane Dress Shoes" the head is Shoes (drop).
    If only a non-dress noun appears, drop. If only a dress noun, keep.
    If neither, keep (trust upstream loose filter).
    """
    dress_hits = list(_DRESS_RE.finditer(title))
    nondress_hits = list(_NON_DRESS_RE.finditer(title))
    if not nondress_hits:
        return False
    if not dress_hits:
        return True
    last_dress = max(m.end() for m in dress_hits)
    last_nondress = max(m.end() for m in nondress_hits)
    return last_nondress > last_dress

WORKSPACE = Path(os.environ.get('PLAYBOOK_WORKSPACE', os.path.expanduser('~')))
DATA_DIR = WORKSPACE / "data"
RAW_DIR = DATA_DIR / "raw" / "raw"
HLLM_DATASET_DIR = WORKSPACE / "hllm-code" / "dataset"
HLLM_INFO_DIR = WORKSPACE / "hllm-code" / "information"
OUT_DIR = DATA_DIR / "processed"

REVIEWS_PATH = RAW_DIR / "review_categories" / "Clothing_Shoes_and_Jewelry.jsonl"
META_PATH = RAW_DIR / "meta_categories" / "meta_Clothing_Shoes_and_Jewelry.jsonl"


def find_dress_items(meta_path):
    """Identify all dress-related items from metadata."""
    print("Identifying dress items from metadata...")
    dress_items = {}
    t0 = time.time()
    with open(meta_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            cats = str(d.get('categories', []))
            title = str(d.get('title', ''))
            if is_dress(title, cats):
                pid = d['parent_asin']
                desc = d.get('description', '')
                if isinstance(desc, list):
                    desc = ' '.join(str(x) for x in desc)
                features = d.get('features', [])
                if isinstance(features, list):
                    features = ' '.join(str(x) for x in features)

                images = d.get('images', [])
                img_url = ''
                if images and isinstance(images, list):
                    first_img = images[0]
                    if isinstance(first_img, dict):
                        img_url = first_img.get('large', first_img.get('thumb', ''))

                price = d.get('price', None)
                try:
                    price = float(price) if price else None
                except (ValueError, TypeError):
                    price = None

                dress_items[pid] = {
                    'item_id': pid,
                    'title': str(d.get('title', ''))[:512],
                    'description': str(desc)[:512],
                    'price': price,
                    'image_url': str(img_url),
                }
            if (i + 1) % 1_000_000 == 0:
                print(f"  Scanned {i+1:,} items, found {len(dress_items):,} dresses")

    print(f"  Total: {len(dress_items):,} dress items in {time.time()-t0:.0f}s")
    return dress_items


def load_dress_reviews(reviews_path, dress_item_ids):
    """Load only reviews for dress items."""
    print(f"Loading dress reviews...")
    rows = []
    t0 = time.time()
    total = 0
    with open(reviews_path) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            if d['parent_asin'] in dress_item_ids:
                rows.append({
                    'user_id': d['user_id'],
                    'item_id': d['parent_asin'],
                    'rating': d['rating'],
                    'timestamp': int(d['timestamp']) // 1000,
                })
            if total % 10_000_000 == 0:
                print(f"  Scanned {total:,}, found {len(rows):,} dress reviews")

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df):,} dress reviews from {total:,} total ({len(df)/total:.1%})")
    return df


def five_core_filter(df, min_count=5):
    """Iteratively filter users and items with < min_count interactions."""
    print(f"5-core filtering...")
    print(f"  Before: {len(df):,} interactions, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= min_count].index)]
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_count].index)]

    print(f"  After: {len(df):,} interactions, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    print(f"  Avg/user: {len(df)/max(df['user_id'].nunique(), 1):.1f}")
    return df


def main():
    t_total = time.time()
    print("=" * 60)
    print("PROCESS AMAZON DRESSES FOR HLLM")
    print("=" * 60)

    # Find dress items
    dress_items = find_dress_items(META_PATH)

    # Load dress reviews
    reviews = load_dress_reviews(REVIEWS_PATH, set(dress_items.keys()))

    # 5-core filter
    filtered = five_core_filter(reviews)

    # Save HLLM format
    HLLM_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    HLLM_INFO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_name = "amazon_dresses"

    # Interactions
    interactions = filtered[['item_id', 'user_id', 'timestamp']].sort_values(['user_id', 'timestamp'])
    interactions.to_csv(HLLM_DATASET_DIR / f"{dataset_name}.csv", index=False)
    print(f"\nSaved interactions: {HLLM_DATASET_DIR / dataset_name}.csv ({len(interactions):,} rows)")

    # Item info (only items in filtered set)
    valid_items = set(filtered['item_id'].unique())
    item_rows = [dress_items[pid] for pid in valid_items if pid in dress_items]
    item_df = pd.DataFrame(item_rows)

    # HLLM format: item_id, title, description
    item_df[['item_id', 'title', 'description']].to_csv(HLLM_INFO_DIR / f"{dataset_name}.csv", index=False)
    print(f"Saved item info: {HLLM_INFO_DIR / dataset_name}.csv ({len(item_df):,} items)")

    # Full metadata for UI
    item_df.to_parquet(OUT_DIR / "dress_metadata.parquet", index=False)
    interactions.to_parquet(OUT_DIR / "dress_interactions.parquet", index=False)

    # Stats
    print(f"\n{'='*60}")
    print(f"FINAL DRESSES DATASET:")
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Users: {interactions['user_id'].nunique():,}")
    print(f"  Items: {interactions['item_id'].nunique():,}")
    print(f"  Avg/user: {len(interactions)/interactions['user_id'].nunique():.1f}")
    has_price = item_df['price'].notna().sum()
    has_image = (item_df['image_url'].str.len() > 0).sum()
    print(f"  Items with price: {has_price:,} ({has_price/len(item_df):.0%})")
    print(f"  Items with image: {has_image:,} ({has_image/len(item_df):.0%})")
    print(f"  Total time: {time.time()-t_total:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
