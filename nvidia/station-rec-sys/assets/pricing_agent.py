"""Dynamic pricing agent for the rec-sys playbook.

Trains a PPO policy that picks per-item price multipliers each day,
evaluated against FixedPrice and AgeMarkdown baselines on a calibrated
inventory simulator. Subcommands:

    smoke   Run baselines on a tiny synthetic catalog and print KPIs.
    train   Train the PPO agent against the simulator.
    eval    Evaluate a trained PPO agent + baselines.
    all     train then eval.

The simulator, demand model, and baselines port code from
`enterprise-retail-demo/src/pricing/` into a single file. Bug fixes from
the prior DQN implementation are applied in the PPO section.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# torch is imported lazily inside the training/eval entry points so the
# `smoke` subcommand can run on environments without GPU drivers.


# ────────────────────────────────────────────────────────────────────
# Pricing config — embedded constants (was configs/pricing.yaml in the
# upstream repo). Inlined here so the playbook stays a single file.
# ────────────────────────────────────────────────────────────────────

PRICING_CONFIG: Dict[str, Any] = {
    "simulator": {
        "horizon_days": 14,
        "num_shoppers_per_day": 10_000,
        "replenish_interval": 7,
        "replenish_fraction": 0.5,
    },
    "inventory": {
        "min_initial": 50,
        "max_initial": 200,
        "cost_fraction_min": 0.40,
        "cost_fraction_max": 0.60,
        "holding_cost_per_unit_day": 0.002,  # ≈ 50%/yr
        "stockout_penalty": 5.0,
    },
    "demand_model": {
        # logit(p) = alpha + beta·rec + gamma·season − epsilon·Δp/p₀ − delta·stockout
        # Categories are derived from price tier on the real Amazon Dresses catalog
        # (luxury = top third by price, midrange = middle, budget = bottom).
        # Elasticity ranges follow Tellis (1988) "The Price Elasticity of Selective
        # Demand": apparel ~2.0, luxury/premium <1.0, budget/clearance 3–4.
        "seasonal_amplitude": 0.2,
        "rec_score_weight": 1.0,
        "stockout_delta": 10.0,
        "elasticity": {
            "luxury":   [0.5, 1.0],   # premium buyers less price-sensitive
            "midrange": [1.5, 2.5],   # standard apparel elasticity
            "budget":   [2.5, 4.0],   # budget shoppers highly price-sensitive
            "default":  [1.5, 2.5],
        },
        # alpha (pre-sigmoid) calibrated to ~6–15 units/day/item with 10K shoppers
        "base_demand": {
            "luxury":   -7.5,  # ~6 units/day — lower volume
            "midrange": -7.0,  # ~9 units/day
            "budget":   -6.5,  # ~15 units/day — volume play
            "default":  -7.0,
        },
    },
    # 9 multipliers covering -40% to +25%. Replaces the upstream [0.85, 1.05]
    # range that couldn't reach the deep markdowns aging stock needs.
    "price_grid": {
        "multipliers": [0.60, 0.70, 0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.25],
    },
}

CATEGORIES: Tuple[str, ...] = ("luxury", "midrange", "budget")


# ────────────────────────────────────────────────────────────────────
# Demand model
# ────────────────────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class DemandModel:
    """Vectorized log-linear demand model.

    Purchase probability per item:

        sigmoid( alpha_cat + beta·rec − epsilon_cat·(p−p₀)/p₀
                 + gamma·season(day) − delta·I(inventory=0) )

    Elasticity is set to the midpoint of each category's configured range
    (deterministic for reproducibility).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.elasticity_ranges = config.get("elasticity", {})
        self.default_elasticity = self.elasticity_ranges.get("default", [1.5, 2.5])
        self.base_demand = config.get("base_demand", {})
        self.default_base_demand = self.base_demand.get("default", -7.0)
        self.seasonal_amplitude = config.get("seasonal_amplitude", 0.2)
        self.rec_score_weight = config.get("rec_score_weight", 1.0)
        self.stockout_delta = config.get("stockout_delta", 10.0)

    def _alpha(self, categories: np.ndarray) -> np.ndarray:
        out = np.full(len(categories), self.default_base_demand, dtype=np.float64)
        for cat, val in self.base_demand.items():
            if cat == "default":
                continue
            out[categories == cat] = val
        return out

    def _epsilon(self, categories: np.ndarray) -> np.ndarray:
        default_mid = float(np.mean(self.default_elasticity))
        out = np.full(len(categories), default_mid, dtype=np.float64)
        for cat, (lo, hi) in self.elasticity_ranges.items():
            if cat == "default":
                continue
            out[categories == cat] = 0.5 * (lo + hi)
        return out

    @staticmethod
    def season_factor(day: int) -> float:
        """Sinusoidal weekly pattern, peaks Saturday (weekday 5)."""
        return float(np.sin(2.0 * np.pi * ((day % 7) - 2) / 7.0))

    def purchase_probability(
        self,
        categories: np.ndarray,
        inventories: np.ndarray,
        base_prices: np.ndarray,
        prices: np.ndarray,
        day: int,
        rec_scores: np.ndarray,
    ) -> np.ndarray:
        alpha = self._alpha(categories)
        epsilon = self._epsilon(categories)
        price_ratio = np.where(
            base_prices > 0, (prices - base_prices) / base_prices, 0.0
        )
        stockout = (inventories <= 0).astype(np.float64)
        logit = (
            alpha
            + self.rec_score_weight * rec_scores
            + self.seasonal_amplitude * self.season_factor(day)
            - epsilon * price_ratio
            - self.stockout_delta * stockout
        )
        return _sigmoid(logit)

    def expected_demand(
        self,
        categories: np.ndarray,
        inventories: np.ndarray,
        base_prices: np.ndarray,
        prices: np.ndarray,
        day: int,
        rec_scores: np.ndarray,
        num_shoppers: int,
    ) -> np.ndarray:
        prob = self.purchase_probability(
            categories, inventories, base_prices, prices, day, rec_scores
        )
        return num_shoppers * prob


# ────────────────────────────────────────────────────────────────────
# Inventory state
# ────────────────────────────────────────────────────────────────────


class InventoryState:
    """Per-item inventory + pricing state. All arrays shape (n_items,)."""

    def __init__(
        self,
        inventories: np.ndarray,
        base_prices: np.ndarray,
        unit_costs: np.ndarray,
        current_prices: np.ndarray,
        days_in_stock: np.ndarray,
        categories: np.ndarray,
        rec_scores: np.ndarray,
    ) -> None:
        self.inventories = inventories.astype(np.float64)
        self.base_prices = base_prices.astype(np.float64)
        self.unit_costs = unit_costs.astype(np.float64)
        self.current_prices = current_prices.astype(np.float64)
        self.days_in_stock = days_in_stock.astype(np.float64)
        self.categories = np.asarray(categories)
        self.rec_scores = rec_scores.astype(np.float64)
        self._initial_inventories = self.inventories.copy()

    @property
    def n_items(self) -> int:
        return len(self.inventories)

    def copy(self) -> "InventoryState":
        return InventoryState(
            inventories=self.inventories.copy(),
            base_prices=self.base_prices.copy(),
            unit_costs=self.unit_costs.copy(),
            current_prices=self.current_prices.copy(),
            days_in_stock=self.days_in_stock.copy(),
            categories=self.categories.copy(),
            rec_scores=self.rec_scores.copy(),
        )

    @classmethod
    def initialize(
        cls,
        item_features: pd.DataFrame,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> "InventoryState":
        """Build a starting state from an item-features DataFrame.

        Expected columns: ``avg_price`` (float), optional ``category`` (str)
        and ``rec_score`` (float). Missing columns are synthesized.
        """
        inv_cfg = config.get("inventory", {})
        rng = np.random.default_rng(seed)
        n = len(item_features)

        if "avg_price" in item_features.columns:
            base_prices = item_features["avg_price"].to_numpy(dtype=np.float64)
            median = np.nanmedian(base_prices[base_prices > 0]) if (base_prices > 0).any() else 50.0
            base_prices = np.where(
                np.isnan(base_prices) | (base_prices <= 0), median, base_prices
            )
        else:
            base_prices = rng.uniform(5.0, 100.0, size=n)

        cf_lo = inv_cfg.get("cost_fraction_min", 0.40)
        cf_hi = inv_cfg.get("cost_fraction_max", 0.60)
        unit_costs = base_prices * rng.uniform(cf_lo, cf_hi, size=n)

        inv_lo = inv_cfg.get("min_initial", 50)
        inv_hi = inv_cfg.get("max_initial", 200)
        inventories = rng.integers(inv_lo, inv_hi + 1, size=n).astype(np.float64)

        if "category" in item_features.columns:
            categories = item_features["category"].to_numpy(dtype=str)
        else:
            categories = np.array([CATEGORIES[i % len(CATEGORIES)] for i in range(n)])

        if "rec_score" in item_features.columns:
            rec_scores = item_features["rec_score"].to_numpy(dtype=np.float64)
        else:
            rec_scores = rng.uniform(0.0, 1.0, size=n)

        return cls(
            inventories=inventories,
            base_prices=base_prices,
            unit_costs=unit_costs,
            current_prices=base_prices.copy(),
            days_in_stock=np.zeros(n, dtype=np.float64),
            categories=categories,
            rec_scores=rec_scores,
        )


# ────────────────────────────────────────────────────────────────────
# Simulator + result tracking
# ────────────────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    daily_revenue: List[float] = field(default_factory=list)
    daily_margin: List[float] = field(default_factory=list)
    daily_units_sold: List[float] = field(default_factory=list)
    daily_stockout_count: List[int] = field(default_factory=list)
    n_items: int = 0
    horizon_days: int = 0
    initial_inventories: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def total_revenue(self) -> float:
        return float(np.sum(self.daily_revenue))

    @property
    def total_margin(self) -> float:
        return float(np.sum(self.daily_margin))

    @property
    def avg_stockout_rate(self) -> float:
        if self.n_items == 0 or self.horizon_days == 0:
            return 0.0
        return float(np.mean(self.daily_stockout_count)) / self.n_items

    @property
    def sell_through_rate(self) -> float:
        total_initial = float(np.sum(self.initial_inventories))
        return (float(np.sum(self.daily_units_sold)) / total_initial) if total_initial > 0 else 0.0


class Simulator:
    """Day-by-day inventory + pricing simulator with weekly replenishment."""

    def __init__(self, demand_model: DemandModel, config: Dict[str, Any]) -> None:
        self.demand_model = demand_model
        sim_cfg = config.get("simulator", {})
        inv_cfg = config.get("inventory", {})
        self.num_shoppers = sim_cfg.get("num_shoppers_per_day", 10_000)
        self.replenish_interval = sim_cfg.get("replenish_interval", 7)
        self.replenish_fraction = sim_cfg.get("replenish_fraction", 0.5)
        self.holding_cost = inv_cfg.get("holding_cost_per_unit_day", 0.002)

    def run(
        self,
        initial_state: InventoryState,
        policy: Any,
        horizon_days: int,
        seed: int = 123,
    ) -> SimulationResult:
        state = initial_state.copy()
        result = SimulationResult(
            n_items=state.n_items,
            horizon_days=horizon_days,
            initial_inventories=state._initial_inventories.copy(),
        )
        rng = np.random.default_rng(seed)

        for day in range(horizon_days):
            try:
                prices = policy.select_prices(state, day=day)
            except TypeError:
                prices = policy.select_prices(state)
            state.current_prices = prices

            expected = self.demand_model.expected_demand(
                categories=state.categories,
                inventories=state.inventories,
                base_prices=state.base_prices,
                prices=prices,
                day=day,
                rec_scores=state.rec_scores,
                num_shoppers=self.num_shoppers,
            )
            realised = rng.poisson(lam=np.clip(expected, 0, None)).astype(np.float64)
            sold = np.minimum(realised, state.inventories)

            day_revenue = float(np.sum(prices * sold))
            day_cogs = float(np.sum(state.unit_costs * sold))
            day_holding = float(
                self.holding_cost * np.sum(state.inventories * state.base_prices)
            )
            day_margin = day_revenue - day_cogs - day_holding

            state.inventories -= sold
            state.days_in_stock = np.where(
                state.inventories > 0,
                state.days_in_stock + 1,
                state.days_in_stock,
            )

            result.daily_revenue.append(day_revenue)
            result.daily_margin.append(day_margin)
            result.daily_units_sold.append(float(np.sum(sold)))
            result.daily_stockout_count.append(int(np.sum(state.inventories <= 0)))

            if (
                self.replenish_interval > 0
                and (day + 1) % self.replenish_interval == 0
            ):
                replenish_level = state._initial_inventories * self.replenish_fraction
                state.inventories = np.maximum(state.inventories, replenish_level)

        return result


# ────────────────────────────────────────────────────────────────────
# Baseline policies
# ────────────────────────────────────────────────────────────────────


class FixedPrice:
    """Always charge the base price — control policy."""

    name = "FixedPrice"

    def select_prices(self, state: InventoryState, day: int = 0) -> np.ndarray:
        return state.base_prices.copy()


class AgeMarkdown:
    """Markdown by ``weekly_discount`` per week of dwell, floored at -50%."""

    name = "AgeMarkdown"

    def __init__(self, weekly_discount: float = 0.05) -> None:
        self.weekly_discount = weekly_discount

    def select_prices(self, state: InventoryState, day: int = 0) -> np.ndarray:
        weeks = state.days_in_stock // 7
        discount = np.clip(self.weekly_discount * weeks, 0.0, 0.50)
        return state.base_prices * (1.0 - discount)


# ────────────────────────────────────────────────────────────────────
# Catalog loaders
# ────────────────────────────────────────────────────────────────────


def _derive_price_tier(prices: np.ndarray) -> np.ndarray:
    """Split items into 3 elasticity-defining tiers by price terciles."""
    valid = prices[prices > 0]
    if len(valid) < 3:
        return np.full(len(prices), "midrange", dtype=object)
    q33, q67 = np.percentile(valid, [100 / 3, 200 / 3])
    tiers = np.where(prices < q33, "budget",
              np.where(prices < q67, "midrange", "luxury"))
    return tiers.astype(object)


def _derive_popularity_rec_score(item_ids: np.ndarray, interactions_path: Path) -> np.ndarray:
    """Per-item popularity in [0, 1], from interaction counts. sqrt-flattened
    so a handful of mega-popular items don't dominate the signal."""
    if not interactions_path.exists():
        return np.full(len(item_ids), 0.5)
    inter = pd.read_parquet(interactions_path, columns=["item_id"])
    counts = inter.groupby("item_id").size()
    item_counts = pd.Series(item_ids).map(counts).fillna(0).to_numpy(dtype=np.float64)
    if item_counts.max() == 0:
        return np.full(len(item_ids), 0.5)
    return np.sqrt(item_counts / item_counts.max())


def load_amazon_dresses_catalog(n_items: int, seed: int) -> pd.DataFrame:
    """Load the real Amazon Dresses catalog produced by the playbook's
    `prepare_data.py`, with derived price-tier categories and popularity-
    based rec_score signals.

    Sample to ``n_items`` for tractable training; pass ``n_items <= 0`` to
    use the full catalog (~14k items after dropping missing prices).
    """
    workspace = workspace_root()
    meta_path = workspace / "data" / "processed" / "dress_metadata.parquet"
    inter_path = workspace / "data" / "processed" / "dress_interactions.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Amazon Dresses metadata not found at {meta_path}.\n"
            f"Run `bash assets/setup.sh` to download/prepare the dataset, "
            f"or pass `--synthetic` to use a generated catalog."
        )
    df = pd.read_parquet(meta_path)
    # Drop items with missing or non-positive prices (~2k of 16k in the dataset).
    df = df[df["price"].notna() & (df["price"] > 0)].reset_index(drop=True)
    if n_items > 0 and n_items < len(df):
        df = df.sample(n=n_items, random_state=seed).reset_index(drop=True)
    df["category"] = _derive_price_tier(df["price"].to_numpy())
    df["rec_score"] = _derive_popularity_rec_score(df["item_id"].to_numpy(), inter_path)
    df = df.rename(columns={"price": "avg_price"})
    return df[["item_id", "category", "avg_price", "rec_score"]]


def build_synthetic_catalog(n_items: int = 100, seed: int = 0) -> pd.DataFrame:
    """A tiny synthetic catalog for the `smoke` subcommand and as a
    `--synthetic` fallback when the real dataset isn't available.

    Uses lognormal prices (median ~$33, matching the Amazon Dresses
    median) and the same luxury/midrange/budget tier scheme so the
    simulator and demand model behave identically across synthetic and
    real catalogs.
    """
    rng = np.random.default_rng(seed)
    prices = np.round(rng.lognormal(mean=3.5, sigma=0.6, size=n_items), 2)
    rec_scores = rng.beta(2.0, 5.0, size=n_items)
    cats = _derive_price_tier(prices)
    return pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "category": cats,
            "avg_price": prices,
            "rec_score": rec_scores,
        }
    )


def load_catalog(n_items: int, seed: int, synthetic: bool) -> pd.DataFrame:
    """Top-level catalog loader: real Amazon Dresses by default, synthetic
    if ``--synthetic`` was passed or the real data is unavailable."""
    if synthetic:
        return build_synthetic_catalog(n_items=n_items if n_items > 0 else 1000, seed=seed)
    return load_amazon_dresses_catalog(n_items=n_items, seed=seed)


# ────────────────────────────────────────────────────────────────────
# Eval helpers
# ────────────────────────────────────────────────────────────────────


def run_policy(
    policy: Any,
    initial_state: InventoryState,
    simulator: Simulator,
    horizon_days: int,
    seed: int = 123,
) -> Tuple[str, SimulationResult, float]:
    name = getattr(policy, "name", type(policy).__name__)
    t0 = time.perf_counter()
    result = simulator.run(initial_state, policy, horizon_days, seed=seed)
    return name, result, time.perf_counter() - t0


def format_kpi_row(name: str, result: SimulationResult, elapsed: float, baseline_rev: float | None) -> str:
    rev = result.total_revenue
    lift = (rev / baseline_rev) if (baseline_rev and baseline_rev > 0) else 1.0
    return (
        f"  {name:18s} | Rev: {rev:10.2f} ({lift:.2f}x) | "
        f"Margin: {result.total_margin:10.2f} | "
        f"Stockout: {result.avg_stockout_rate:5.1%} | "
        f"Sell-through: {result.sell_through_rate:5.1%} | "
        f"{elapsed*1000:5.0f}ms"
    )


def print_kpi_table(rows: List[Tuple[str, SimulationResult, float]]) -> None:
    if not rows:
        return
    baseline_rev = rows[0][1].total_revenue
    print("-" * 110)
    for name, res, elapsed in rows:
        print(format_kpi_row(name, res, elapsed, baseline_rev))
    print("-" * 110)


# ────────────────────────────────────────────────────────────────────
# PPO agent (torch) — discrete action over price multipliers,
# shared MLP backbone, vectorized simulator envs.
# ────────────────────────────────────────────────────────────────────


CATEGORY_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
N_CATEGORIES: int = len(CATEGORIES)
N_STATE_FEATURES: int = 8 + N_CATEGORIES  # 8 numeric + 3 one-hot


def encode_state_per_item(
    state: InventoryState,
    day: int,
    horizon: int,
    price_norm: float,
    inv_norm: float,
) -> np.ndarray:
    """Return a (n_items, N_STATE_FEATURES) float32 array.

    Features per item (in order):
        0  inventory_ratio (current / initial)
        1  days_in_stock / horizon
        2  day / horizon
        3  sin(2π · day_of_week / 7)
        4  cos(2π · day_of_week / 7)
        5  log1p(base_price) / log1p(price_norm)
        6  log1p(inventory)  / log1p(inv_norm)
        7  rec_score
        8..10  category one-hot (fashion, basics, seasonal)
    """
    n = state.n_items
    initial = state._initial_inventories
    inv_ratio = np.where(initial > 0, state.inventories / initial, 0.0)
    days_norm = state.days_in_stock / max(horizon, 1)
    day_norm = np.full(n, day / max(horizon, 1))
    day_of_week = day % 7
    dow_sin = np.full(n, np.sin(2.0 * np.pi * day_of_week / 7.0))
    dow_cos = np.full(n, np.cos(2.0 * np.pi * day_of_week / 7.0))
    bp = np.log1p(state.base_prices) / np.log1p(max(price_norm, 1.0))
    inv = np.log1p(state.inventories) / np.log1p(max(inv_norm, 1.0))
    rec = state.rec_scores

    cat_onehot = np.zeros((n, N_CATEGORIES), dtype=np.float64)
    for i, cat in enumerate(state.categories):
        idx = CATEGORY_TO_IDX.get(str(cat), 0)
        cat_onehot[i, idx] = 1.0

    return np.column_stack(
        [inv_ratio, days_norm, day_norm, dow_sin, dow_cos, bp, inv, rec, cat_onehot]
    ).astype(np.float32)


class PricingEnv:
    """Gym-like wrapper exposing the simulator one day at a time.

    ``reset()`` returns the encoded state. ``step(actions)`` advances one
    day and returns (next_state, per_item_reward, done, info).
    Per-item reward is the per-item margin contribution to total margin,
    so summing over items recovers the eval-metric exactly (no
    reward-eval drift).
    """

    def __init__(
        self,
        initial_state: InventoryState,
        simulator: Simulator,
        multipliers: np.ndarray,
        horizon: int,
        seed: int,
    ) -> None:
        self.initial_state = initial_state
        self.simulator = simulator
        self.multipliers = multipliers
        self.horizon = horizon
        self.seed = seed
        self._price_norm = float(self.initial_state.base_prices.max())
        self._inv_norm = float(self.initial_state._initial_inventories.max())
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = self.initial_state.copy()
        self.day = 0
        self.rng = np.random.default_rng(self.seed)
        return encode_state_per_item(
            self.state, self.day, self.horizon, self._price_norm, self._inv_norm
        )

    def step(self, action_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, float]]:
        prices = self.state.base_prices * self.multipliers[action_idx]
        self.state.current_prices = prices

        expected = self.simulator.demand_model.expected_demand(
            categories=self.state.categories,
            inventories=self.state.inventories,
            base_prices=self.state.base_prices,
            prices=prices,
            day=self.day,
            rec_scores=self.state.rec_scores,
            num_shoppers=self.simulator.num_shoppers,
        )
        realised = self.rng.poisson(lam=np.clip(expected, 0, None)).astype(np.float64)
        sold = np.minimum(realised, self.state.inventories)

        revenue = prices * sold
        cogs = self.state.unit_costs * sold
        holding = self.simulator.holding_cost * self.state.inventories * self.state.base_prices
        reward_per_item = (revenue - cogs - holding).astype(np.float32)

        self.state.inventories -= sold
        self.state.days_in_stock = np.where(
            self.state.inventories > 0,
            self.state.days_in_stock + 1,
            self.state.days_in_stock,
        )
        self.day += 1
        if (
            self.simulator.replenish_interval > 0
            and self.day % self.simulator.replenish_interval == 0
        ):
            replenish_level = self.state._initial_inventories * self.simulator.replenish_fraction
            self.state.inventories = np.maximum(self.state.inventories, replenish_level)

        done = self.day >= self.horizon
        next_obs = encode_state_per_item(
            self.state, self.day, self.horizon, self._price_norm, self._inv_norm
        )
        info = {
            "total_margin": float(reward_per_item.sum()),
            "total_revenue": float(revenue.sum()),
        }
        return next_obs, reward_per_item, done, info


# ────────────────────────────────────────────────────────────────────
# Network
# ────────────────────────────────────────────────────────────────────


def _build_actor_critic(n_actions: int, hidden: int = 256):
    """Return an ActorCritic torch module. Imported lazily."""
    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(N_STATE_FEATURES, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.actor = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)

        def forward(self, x):
            h = self.backbone(x)
            return self.actor(h), self.critic(h).squeeze(-1)

    return ActorCritic()


# ────────────────────────────────────────────────────────────────────
# PPO Trainer
# ────────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    n_iters: int = 200
    n_envs: int = 16
    horizon: int = 14
    n_items: int = 1000
    n_epochs: int = 4
    minibatch_size: int = 4096
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.005
    max_grad_norm: float = 0.5
    reward_scale: float = 100.0
    value_clip_eps: float = 0.2
    device: str = "auto"
    seed: int = 0


def _compute_gae(
    rewards: np.ndarray,  # (T, K, N)
    values: np.ndarray,   # (T+1, K, N)
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation per item-trajectory.

    Treats each (env, item) pair as an independent length-T trajectory.
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1:], dtype=np.float32)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


class PPOTrainer:
    """Vectorized PPO trainer for the pricing env."""

    def __init__(self, cfg: PPOConfig, catalog: pd.DataFrame, multipliers: np.ndarray):
        import torch

        self.cfg = cfg
        self.multipliers = multipliers
        self.n_actions = len(multipliers)
        self.device = self._resolve_device(cfg.device)

        # Build K parallel envs with different seeds for diversity.
        self.envs: List[PricingEnv] = []
        for k in range(cfg.n_envs):
            initial_state = InventoryState.initialize(catalog, PRICING_CONFIG, seed=cfg.seed + k)
            demand = DemandModel(PRICING_CONFIG["demand_model"])
            simulator = Simulator(demand, PRICING_CONFIG)
            self.envs.append(
                PricingEnv(initial_state, simulator, multipliers, cfg.horizon, seed=cfg.seed + k)
            )

        self.net = _build_actor_critic(self.n_actions).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.train_curve: List[float] = []  # mean total revenue per iteration

    @staticmethod
    def _resolve_device(s: str) -> str:
        import torch
        if s == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return s

    def _act(self, obs_kn: "Any", greedy: bool = False):
        """obs_kn: (K, N, F) torch tensor. Returns actions, log_probs, values, entropy."""
        import torch
        logits, value = self.net(obs_kn)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1).gather(
            -1, action.unsqueeze(-1)
        ).squeeze(-1)
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        return action, log_prob, value, entropy

    def collect_rollout(self):
        """Run K parallel envs for `horizon` steps. Returns dict of tensors."""
        import torch
        cfg = self.cfg
        n_items = self.envs[0].state.n_items
        obs_buf = np.zeros((cfg.horizon, cfg.n_envs, n_items, N_STATE_FEATURES), dtype=np.float32)
        act_buf = np.zeros((cfg.horizon, cfg.n_envs, n_items), dtype=np.int64)
        logp_buf = np.zeros((cfg.horizon, cfg.n_envs, n_items), dtype=np.float32)
        val_buf = np.zeros((cfg.horizon + 1, cfg.n_envs, n_items), dtype=np.float32)
        rew_buf = np.zeros((cfg.horizon, cfg.n_envs, n_items), dtype=np.float32)

        obs = np.stack([env.reset() for env in self.envs], axis=0)  # (K, N, F)
        episode_returns = np.zeros(cfg.n_envs, dtype=np.float64)
        episode_revenues = np.zeros(cfg.n_envs, dtype=np.float64)

        for t in range(cfg.horizon):
            obs_t = torch.from_numpy(obs).to(self.device)  # (K, N, F)
            with torch.no_grad():
                action, log_prob, value, _ = self._act(obs_t, greedy=False)
            action_np = action.cpu().numpy()

            obs_buf[t] = obs
            act_buf[t] = action_np
            logp_buf[t] = log_prob.cpu().numpy()
            val_buf[t] = value.cpu().numpy()

            next_obs_list = []
            for k, env in enumerate(self.envs):
                next_o, reward, _done, info = env.step(action_np[k])
                rew_buf[t, k] = reward / cfg.reward_scale  # scale rewards for stable critic
                episode_returns[k] += info["total_margin"]
                episode_revenues[k] += info["total_revenue"]
                next_obs_list.append(next_o)
            obs = np.stack(next_obs_list, axis=0)

        # Bootstrap value for last state (used by GAE)
        with torch.no_grad():
            obs_T = torch.from_numpy(obs).to(self.device)
            _, _, last_value, _ = self._act(obs_T, greedy=False)
        val_buf[cfg.horizon] = last_value.cpu().numpy()

        advantages, returns = _compute_gae(rew_buf, val_buf, cfg.gamma, cfg.gae_lambda)

        return {
            "obs": obs_buf,
            "actions": act_buf,
            "log_probs": logp_buf,
            "values": val_buf[:-1],  # old values for clipped value loss
            "advantages": advantages,
            "returns": returns,
            "episode_returns": episode_returns,
            "episode_revenues": episode_revenues,
        }

    def update(self, rollout: Dict[str, np.ndarray], entropy_coef: float) -> Dict[str, float]:
        """Run PPO epochs on the collected rollout. Returns loss diagnostics."""
        import torch

        cfg = self.cfg
        obs = torch.from_numpy(rollout["obs"]).to(self.device).reshape(-1, N_STATE_FEATURES)
        actions = torch.from_numpy(rollout["actions"]).to(self.device).reshape(-1)
        old_log_probs = torch.from_numpy(rollout["log_probs"]).to(self.device).reshape(-1)
        old_values = torch.from_numpy(rollout["values"]).to(self.device).reshape(-1)
        advantages = torch.from_numpy(rollout["advantages"]).to(self.device).reshape(-1)
        returns = torch.from_numpy(rollout["returns"]).to(self.device).reshape(-1)
        # Normalize advantages for stable gradients
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = obs.shape[0]
        idx = torch.arange(N, device=self.device)

        policy_losses, value_losses, entropies = [], [], []
        for _epoch in range(cfg.n_epochs):
            perm = idx[torch.randperm(N, device=self.device)]
            for start in range(0, N, cfg.minibatch_size):
                mb = perm[start : start + cfg.minibatch_size]
                logits, value = self.net(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_log_prob - old_log_probs[mb]).exp()
                adv = advantages[mb]
                surrogate1 = ratio * adv
                surrogate2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Clipped value loss: prevent value-function divergence after policy converges.
                v_clipped = old_values[mb] + torch.clamp(
                    value - old_values[mb], -cfg.value_clip_eps, cfg.value_clip_eps
                )
                v_loss_unclipped = (value - returns[mb]).pow(2)
                v_loss_clipped = (v_clipped - returns[mb]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optim.step()

                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def train(self, verbose: bool = True) -> List[float]:
        cfg = self.cfg
        if verbose:
            print(
                f"Training PPO: {cfg.n_iters} iters × {cfg.n_envs} envs × "
                f"{cfg.horizon} days × {cfg.n_items} items on {self.device}"
            )
        t0 = time.perf_counter()
        for it in range(cfg.n_iters):
            # Linear anneal of entropy bonus from start → end across training.
            frac = it / max(1, cfg.n_iters - 1)
            entropy_coef = cfg.entropy_coef_start + (cfg.entropy_coef_end - cfg.entropy_coef_start) * frac
            rollout = self.collect_rollout()
            losses = self.update(rollout, entropy_coef=entropy_coef)
            mean_revenue = float(rollout["episode_revenues"].mean())
            mean_margin = float(rollout["episode_returns"].mean())
            self.train_curve.append(mean_revenue)
            if verbose and (it + 1) % max(1, cfg.n_iters // 20) == 0:
                print(
                    f"  iter {it + 1:4d}/{cfg.n_iters} | "
                    f"rev/ep: {mean_revenue:10.0f} | margin/ep: {mean_margin:10.0f} | "
                    f"pi_loss: {losses['policy_loss']:+.3f} | "
                    f"v_loss: {losses['value_loss']:.3f} | "
                    f"H: {losses['entropy']:.3f} | "
                    f"ent_c: {entropy_coef:.3f}"
                )
        if verbose:
            print(f"Training complete in {time.perf_counter() - t0:.1f}s")
        return self.train_curve


class PPOPolicy:
    """Inference-time wrapper exposing the PricingPolicy interface."""

    name = "PPO"

    def __init__(self, net, multipliers: np.ndarray, device: str, horizon: int,
                 price_norm: float, inv_norm: float, greedy: bool = True) -> None:
        self.net = net
        self.multipliers = multipliers
        self.device = device
        self.horizon = horizon
        self.price_norm = price_norm
        self.inv_norm = inv_norm
        self.greedy = greedy
        self.net.eval()

    def select_prices(self, state: InventoryState, day: int = 0) -> np.ndarray:
        import torch
        obs = encode_state_per_item(state, day, self.horizon, self.price_norm, self.inv_norm)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(self.device)
            logits, _ = self.net(obs_t)
            if self.greedy:
                action = logits.argmax(dim=-1).cpu().numpy()
            else:
                action = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        return state.base_prices * self.multipliers[action]


# ────────────────────────────────────────────────────────────────────
# Workspace + persistence helpers
# ────────────────────────────────────────────────────────────────────


def workspace_root() -> Path:
    return Path(os.environ.get("PLAYBOOK_WORKSPACE", str(Path.home())))


def model_dir() -> Path:
    return workspace_root() / "models" / "pricing_ppo"


def processed_dir() -> Path:
    return workspace_root() / "data" / "processed"


def save_checkpoint(
    net,
    cfg: PPOConfig,
    multipliers: np.ndarray,
    catalog_meta: Dict[str, Any],
    train_curve: List[float],
    path: Path,
) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": cfg.__dict__,
            "multipliers": multipliers.tolist(),
            "catalog_meta": catalog_meta,
            "train_curve": train_curve,
        },
        path,
    )


def load_checkpoint(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def save_training_curve_png(train_curve: List[float], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available, skipping {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_curve, color="#76b900", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode revenue")
    ax.set_title("PPO training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run baselines on a tiny synthetic catalog and print KPIs."""
    print(f"Running pricing smoke test ({args.n_items} items, {args.horizon} days)…")
    catalog = build_synthetic_catalog(n_items=args.n_items, seed=args.seed)
    print(
        f"Catalog: {len(catalog)} items, categories: "
        f"{dict(zip(*np.unique(catalog['category'], return_counts=True)))}"
    )

    demand = DemandModel(PRICING_CONFIG["demand_model"])
    simulator = Simulator(demand, PRICING_CONFIG)
    initial_state = InventoryState.initialize(catalog, PRICING_CONFIG, seed=args.seed)

    policies = [FixedPrice(), AgeMarkdown(weekly_discount=0.05)]
    rows = [run_policy(p, initial_state, simulator, args.horizon, seed=args.seed) for p in policies]
    print_kpi_table(rows)
    return 0


def _ppo_config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        n_iters=args.n_iters,
        n_envs=args.n_envs,
        horizon=args.horizon,
        n_items=args.n_items,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )


def _print_catalog_summary(catalog: pd.DataFrame, source: str) -> None:
    cat_counts = catalog["category"].value_counts().to_dict()
    price_med = float(catalog["avg_price"].median())
    print(
        f"Catalog [{source}]: {len(catalog)} items | "
        f"price median ${price_med:.2f} | tiers: {cat_counts}"
    )


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _ppo_config_from_args(args)
    catalog = load_catalog(n_items=cfg.n_items, seed=cfg.seed, synthetic=args.synthetic)
    source = "synthetic" if args.synthetic else "Amazon Dresses"
    multipliers = np.asarray(PRICING_CONFIG["price_grid"]["multipliers"], dtype=np.float64)
    _print_catalog_summary(catalog, source)
    print(f"Multipliers: {multipliers.tolist()}, horizon: {cfg.horizon}d")

    trainer = PPOTrainer(cfg, catalog, multipliers)
    trainer.train(verbose=True)

    ckpt_path = model_dir() / "policy.pt"
    catalog_meta = {
        "n_items": len(catalog),
        "seed": cfg.seed,
        "synthetic": args.synthetic,
        "source": source,
        "price_norm": float(catalog["avg_price"].max()),
    }
    save_checkpoint(trainer.net, cfg, multipliers, catalog_meta, trainer.train_curve, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")

    curve_path = processed_dir() / "pricing_training_curve.png"
    save_training_curve_png(trainer.train_curve, curve_path)
    print(f"Saved training curve → {curve_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    ckpt_path = model_dir() / "policy.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run `train` first.", file=sys.stderr)
        return 2

    ckpt = load_checkpoint(ckpt_path)
    cfg_dict = ckpt["config"]
    multipliers = np.asarray(ckpt["multipliers"], dtype=np.float64)
    catalog_meta = ckpt.get("catalog_meta", {})
    synthetic = bool(catalog_meta.get("synthetic", False))
    n_items = int(catalog_meta.get("n_items", cfg_dict.get("n_items", 1000)))
    catalog = load_catalog(n_items=n_items, seed=args.seed, synthetic=synthetic)
    source = catalog_meta.get("source", "synthetic" if synthetic else "Amazon Dresses")
    _print_catalog_summary(catalog, source)

    demand = DemandModel(PRICING_CONFIG["demand_model"])
    simulator = Simulator(demand, PRICING_CONFIG)
    initial_state = InventoryState.initialize(catalog, PRICING_CONFIG, seed=args.seed)

    device = PPOTrainer._resolve_device(args.device)
    net = _build_actor_critic(len(multipliers)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    ppo_policy = PPOPolicy(
        net=net,
        multipliers=multipliers,
        device=device,
        horizon=cfg_dict["horizon"],
        price_norm=float(initial_state.base_prices.max()),
        inv_norm=float(initial_state._initial_inventories.max()),
        greedy=True,
    )

    policies = [FixedPrice(), AgeMarkdown(weekly_discount=0.05), ppo_policy]
    print(f"\nEvaluating ({cfg_dict['n_items']} items, {cfg_dict['horizon']} days, device={device}):")
    rows = [run_policy(p, initial_state, simulator, cfg_dict["horizon"], seed=args.seed) for p in policies]
    print_kpi_table(rows)

    # Save eval results
    eval_path = processed_dir() / "pricing_eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_rev = rows[0][1].total_revenue
    summary = {
        "n_items": cfg_dict["n_items"],
        "horizon": cfg_dict["horizon"],
        "device": device,
        "seed": args.seed,
        "policies": [
            {
                "name": name,
                "total_revenue": res.total_revenue,
                "total_margin": res.total_margin,
                "avg_stockout_rate": res.avg_stockout_rate,
                "sell_through_rate": res.sell_through_rate,
                "revenue_lift_vs_fixed_price": (res.total_revenue / baseline_rev) if baseline_rev > 0 else 1.0,
                "elapsed_s": elapsed,
            }
            for name, res, elapsed in rows
        ],
    }
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved eval summary → {eval_path}")
    return 0


def cmd_train_and_eval(args: argparse.Namespace) -> int:
    rc = cmd_train(args)
    if rc != 0:
        return rc
    return cmd_eval(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pricing_agent",
        description="Dynamic pricing agent for the rec-sys playbook.",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_smoke = sub.add_parser("smoke", help="Run baselines on a tiny synthetic catalog.")
    p_smoke.add_argument("--n-items", type=int, default=100)
    p_smoke.add_argument("--horizon", type=int, default=14)
    p_smoke.add_argument("--seed", type=int, default=0)
    p_smoke.set_defaults(func=cmd_smoke)

    def _add_train_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--n-iters", type=int, default=200)
        p.add_argument("--n-envs", type=int, default=16)
        p.add_argument("--horizon", type=int, default=14)
        p.add_argument(
            "--n-items",
            type=int,
            default=1000,
            help="Sample size from the Amazon Dresses catalog. Pass 0 for the full ~14k items.",
        )
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--synthetic",
            action="store_true",
            help="Use a generated synthetic catalog instead of the real Amazon Dresses data.",
        )

    p_train = sub.add_parser("train", help="Train the PPO agent against the simulator.")
    _add_train_flags(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate trained PPO + baselines.")
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.set_defaults(func=cmd_eval)

    p_tae = sub.add_parser(
        "train_and_eval",
        help="Train the PPO agent, then evaluate vs. baselines. Default if no subcommand is given.",
    )
    _add_train_flags(p_tae)
    p_tae.set_defaults(func=cmd_train_and_eval)

    return parser


_VALID_SUBCOMMANDS = {"smoke", "train", "eval", "train_and_eval"}


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Default to `train_and_eval` if the user didn't pick a subcommand. Letting
    # `--help` through gives them the top-level help; anything else (flags, no
    # args) implies they want the full train→eval flow.
    if not argv or (argv[0] not in _VALID_SUBCOMMANDS and argv[0] not in {"-h", "--help"}):
        argv = ["train_and_eval", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
