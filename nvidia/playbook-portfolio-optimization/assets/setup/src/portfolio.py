# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

"""Portfolio class for managing and analyzing investment portfolios."""

import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Portfolio:
    """
    Portfolio class for managing asset allocations and cash holdings.

    Stores portfolio weights, cash allocation, and provides methods for
    portfolio analysis and visualization.

    Attributes
    ----------
    name : str
        Portfolio identifier
    tickers : list
        Asset symbols/tickers
    weights : np.ndarray
        Portfolio weights for each asset
    cash : float
        Cash allocation (typically 0-1)
    time_range : tuple, optional
        (start_date, end_date) for portfolio period
    """

    def __init__(self, name="", tickers=None, weights=None, cash=0.0, time_range=None):
        """
        Initialize Portfolio with assets, weights, and cash allocation.

        Parameters
        ----------
        name : str, default ""
            Portfolio identifier/name
        tickers : list, optional
            Asset symbols (e.g., ['AAPL', 'MSFT'])
        weights : array-like, optional
            Portfolio weights for each asset
        cash : float, default 0.0
            Cash allocation
        time_range : tuple, optional
            (start_date, end_date) for portfolio period
        """
        self.name = name
        self.tickers = tickers if tickers is not None else []
        self._n_assets = len(self.tickers)
        self.weights = weights if weights is not None else []
        self.cash = float(cash)
        self.time_range = time_range

    def __eq__(self, other_portfolio, atol=1e-3):
        """Check portfolio equality based on weights within tolerance."""
        if isinstance(other_portfolio, Portfolio):
            return np.allclose(self.weights, other_portfolio.weights, atol=atol)
        return False

    def _check_self_financing(self, weights=None, cash=None):
        """Verify that portfolio weights and cash sum to 1.0
        (self-financing constraint)."""
        if weights is None or weights.size == 0:
            weights = self.weights
        if cash is None:
            cash = self.cash

        self_finance = np.sum(weights) + cash

        if np.abs(self_finance - 1) > 1e-3:
            print(f"weights: {np.sum(weights)}; cash: {cash}")
            raise ValueError("Portfolio weights and cash do not sum to 1!")

    def portfolio_from_dict(self, portfolio_name, user_portfolio_dict, cash):
        """
        Create portfolio from user-specified weights dictionary.

        Parameters
        ----------
        portfolio_name : str
            Name for the portfolio
        user_portfolio_dict : dict
            Asset weights as {ticker: weight}
        cash : float
            Cash allocation
        """

        weights = pd.Series(dtype=np.float64, index=self.tickers)

        for ticker, weight in user_portfolio_dict.items():
            ticker = ticker.upper()
            if ticker in self.tickers:
                weights[ticker] = weight
            else:
                raise ValueError("Selected ticker is not available in the dataset!")

        weights = weights.fillna(0).T
        weights = weights.to_numpy()

        self._check_self_financing(weights, cash)

        self.weights = np.array(weights)
        self.cash = float(cash)
        self.name = portfolio_name

    def print_clean(self, cutoff=1e-3, min_percentage=0.0, rounding=3, verbose=False):
        """
        Display clean portfolio allocation with formatting.

        Filters positions based on both cutoff threshold and minimum percentage.
        Only positions meeting both criteria are displayed/returned.

        Parameters
        ----------
        cutoff : float, default 1e-3
            Minimum absolute weight to display (positions below this are
            considered zero).
        min_percentage : float, default 0.0
            Minimum percentage threshold (0-100) for displaying/returning assets.
            Only assets with absolute allocation >= min_percentage% will be included.
            Example: min_percentage=2.0 shows only assets with ≥2% allocation.
        rounding : int, default 3
            Number of decimal places for display.
        verbose : bool, default False
            Whether to print portfolio breakdown.

        Returns
        -------
        tuple
            (clean_portfolio_dict, cash) - Dictionary of significant positions
            and cash amount.
        """
        residual = 0
        clean_ptf_dict = {}
        long_positions = {}
        short_positions = {}

        # Process each position
        for idx, ticker in enumerate(self.tickers):
            value = self.weights[idx]
            if value > cutoff:
                clean_ptf_dict[ticker] = value
                long_positions[ticker] = value
            elif value < -cutoff:
                clean_ptf_dict[ticker] = value
                short_positions[ticker] = value
            else:
                residual += value

        cash = round(self.cash, rounding)

        # Filter positions by minimum percentage threshold
        min_threshold = min_percentage / 100.0  # Convert percentage to decimal
        if min_threshold > 0:
            # Filter clean_ptf_dict to only include positions >= min_percentage%
            clean_ptf_dict = {
                k: v for k, v in clean_ptf_dict.items() if abs(v) >= min_threshold
            }
            # Re-separate long and short positions after filtering
            long_positions = {k: v for k, v in clean_ptf_dict.items() if v > 0}
            short_positions = {k: v for k, v in clean_ptf_dict.items() if v < 0}
            # Filter cash if below threshold
            if abs(cash) < min_threshold:
                cash = 0.0

        if verbose:
            # Portfolio header
            print(f"\nPORTFOLIO: {self.name.upper()}")
            print(f"{'-' * 40}")
            if self.time_range:
                print(f"Period: {self.time_range[0]} to {self.time_range[1]}")

            # Long positions section
            if long_positions:
                print(f"\nLONG POSITIONS ({len(long_positions)} assets)")
                print(f"{'-' * 25}")
                total_long = 0
                for ticker, weight in sorted(
                    long_positions.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"{ticker:8} {weight:>8.{rounding}f} ({weight * 100:>6.2f}%)")
                    total_long += weight
                print(
                    f"{'Total Long':8} {total_long:>8.{rounding}f} "
                    f"({total_long * 100:>6.2f}%)"
                )

            # Short positions section
            if short_positions:
                print(f"\nSHORT POSITIONS ({len(short_positions)} assets)")
                print(f"{'-' * 26}")
                total_short = 0
                for ticker, weight in sorted(
                    short_positions.items(), key=lambda x: x[1]
                ):
                    print(f"{ticker:8} {weight:>8.{rounding}f} ({weight * 100:>6.2f}%)")
                    total_short += weight
                print(
                    f"{'Total Short':8} {total_short:>8.{rounding}f} "
                    f"({total_short * 100:>6.2f}%)"
                )

            # Cash and summary section
            print("\nCASH & SUMMARY")
            print(f"{'-' * 20}")
            print(f"{'Cash':8} {cash:>8.{rounding}f} ({cash * 100:>6.2f}%)")

            if abs(residual) > 1e-6:
                print(
                    f"{'Residual':8} {residual:>8.{rounding}f} "
                    f"({residual * 100:>6.2f}%)"
                )

            # Portfolio totals
            net_equity = sum(clean_ptf_dict.values())
            total_allocation = net_equity + cash + residual
            gross_exposure = sum(abs(w) for w in clean_ptf_dict.values())

            print(
                f"\n{'Net Equity':15} {net_equity:>8.{rounding}f} "
                f"({net_equity * 100:>6.2f}%)"
            )
            print(
                f"{'Total Portfolio':15} {total_allocation:>8.{rounding}f} "
                f"({total_allocation * 100:>6.2f}%)"
            )
            print(
                f"{'Gross Exposure':15} {gross_exposure:>8.{rounding}f} "
                f"({gross_exposure * 100:>6.2f}%)"
            )

            print(f"{'-' * 40}")

        return (clean_ptf_dict, float(cash))

    def calculate_portfolio_expected_return(self, mean):
        """
        Calculate portfolio expected return from asset mean returns.

        Parameters
        ----------
        mean : np.ndarray
            Mean returns for each asset (shape: n_assets)

        Returns
        -------
        float
            Portfolio expected return
        """
        assert (
            mean.shape[0] == self._n_assets
        ), f"Incorrect mean vector size! Expecting: {self._n_assets}."

        return mean @ self.weights

    def calculate_portfolio_variance(self, covariance):
        """
        Calculate portfolio variance from asset covariance matrix.

        Parameters
        ----------
        covariance : np.ndarray
            Asset covariance matrix (shape: n_assets x n_assets)

        Returns
        -------
        float
            Portfolio variance
        """
        assert (
            covariance.shape[0] == self._n_assets
            or covariance.shape[1] != self._n_assets
        ), (
            f"Incorrect covariance size! Expecting: {self._n_assets} by "
            + f"{self._n_assets}."
        )

        return self.weights.T @ covariance @ self.weights

    def plot_portfolio(
        self,
        show_plot=False,
        ax=None,
        title=None,
        figsize=(12, 8),
        style="modern",
        cutoff=1e-3,
        min_percentage=0.0,
        sort_by_weight=True,
        save_path=None,
        dpi=300,
    ):
        """
        Create a portfolio allocation visualization with gradient colors.

        Uses color gradients based on position weights:
        - Long positions: Blue gradient (light to dark based on weight)
        - Short positions: Red gradient (light to dark based on absolute weight)
        - Cash: Yellow (separate category at bottom)

        Only displays assets above the minimum percentage threshold for cleaner plots.

        Parameters
        ----------
        show_plot : bool, default False
            Whether to display the plot immediately.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure.
        title : str, optional
            Custom plot title. If None, auto-generates from portfolio info.
        figsize : tuple, default (12, 8)
            Figure size (width, height) in inches.
        style : str, default "modern"
            Visual style ("modern", "classic", "minimal").
        cutoff : float, default 1e-3
            Minimum weight to display (smaller positions grouped as "Other").
        min_percentage : float, default 0.0
            Minimum percentage threshold (0-100) for displaying assets.
            Only assets with absolute allocation >= min_percentage% will be shown.
            Example: min_percentage=1.0 shows only assets with ≥1% allocation.
        sort_by_weight : bool, default True
            Whether to sort positions by absolute weight (largest first).
        save_path : str, optional
            Path to save the plot. If None, plot is not saved.
        dpi : int, default 300
            Resolution for saved figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        # Color schemes consistent with rebalance plot
        color_schemes = {
            "modern": {
                "long": "#7cd7fe",  # Light blue (frontier)
                "short": "#ff8181",  # Pink/Red (custom)
                "cash": "#fcde7b",  # Purple (assets)
                "background": "#ffffff",
                "grid": "#E0E0E0",
                "text": "#000000",
            }
        }

        colors = color_schemes.get(style, color_schemes["modern"])
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize, dpi=dpi, facecolor=colors["background"]
            )
            ax.set_facecolor(colors["background"])
        else:
            _ = ax.get_figure()  # fig (unused)

        # Get portfolio data (filtering handled by print_clean)
        portfolio_data, filtered_cash = self.print_clean(
            cutoff=cutoff, min_percentage=min_percentage
        )
        cash = filtered_cash

        # Separate long and short positions
        long_positions = {k: v for k, v in portfolio_data.items() if v > 0}
        short_positions = {k: v for k, v in portfolio_data.items() if v < 0}

        # Prepare data for plotting
        all_tickers = []
        all_weights = []
        all_colors = []

        # Sort positions if requested
        if sort_by_weight:
            long_sorted = sorted(
                long_positions.items(), key=lambda x: x[1], reverse=True
            )
            short_sorted = sorted(
                short_positions.items(), key=lambda x: abs(x[1]), reverse=True
            )
        else:
            long_sorted = list(long_positions.items())
            short_sorted = list(short_positions.items())

        # Create color gradients based on weights
        # For long positions: gradient from light to dark blue
        if long_positions:
            max_long = max(long_positions.values())
            min_long = (
                min(long_positions.values())
                if len(long_positions) > 1
                else max_long * 0.1
            )
            long_range = max_long - min_long if max_long != min_long else max_long

            # Create blue gradient colormap
            long_cmap = mcolors.LinearSegmentedColormap.from_list(
                "long_gradient", ["#7cd7fe", "#0046a4"], N=100
            )

        # For short positions: gradient from light to dark red
        if short_positions:
            max_short = max(abs(w) for w in short_positions.values())
            min_short = (
                min(abs(w) for w in short_positions.values())
                if len(short_positions) > 1
                else max_short * 0.1
            )
            short_range = max_short - min_short if max_short != min_short else max_short

            # Create red gradient colormap
            short_cmap = mcolors.LinearSegmentedColormap.from_list(
                "short_gradient", ["#ff8181", "#961515"], N=100
            )

        # Add long positions with gradient colors
        for ticker, weight in long_sorted:
            all_tickers.append(ticker)
            all_weights.append(weight)

            if long_range > 0:
                # Normalize weight to [0, 1] for colormap
                intensity = (weight - min_long) / long_range
                color = long_cmap(intensity)
            else:
                color = long_cmap(0.8)  # Default intensity
            all_colors.append(color)

        # Add short positions with gradient colors
        for ticker, weight in short_sorted:
            all_tickers.append(ticker)
            all_weights.append(weight)

            if short_range > 0:
                # Normalize absolute weight to [0, 1] for colormap
                intensity = (abs(weight) - min_short) / short_range
                color = short_cmap(intensity)
            else:
                color = short_cmap(0.8)  # Default intensity
            all_colors.append(color)

        # Add cash at the bottom as separate category (yellow)
        if abs(cash) > cutoff:
            all_tickers.append("CASH")
            all_weights.append(cash)
            all_colors.append(colors["cash"])  # Gold/Yellow color for cash

        # Create horizontal bar chart with extra space before cash
        cash_gap = 0.8  # Extra space between equity positions and cash
        y_positions = []

        # Calculate positions with gap before cash
        for i, ticker in enumerate(all_tickers):
            if (
                ticker == "CASH" and i > 0
            ):  # Add gap before cash if it's not the only item
                y_positions.append(i + cash_gap)
            else:
                y_positions.append(i)

        _ = ax.barh(  # bars (unused)
            y_positions,
            all_weights,
            color=all_colors,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.8,
        )

        # Customize appearance - reverse y-axis so first items appear at top
        ax.set_yticks(y_positions)
        ax.set_yticklabels(all_tickers, fontsize=9, color=colors["text"])
        ax.invert_yaxis()  # Reverse y-axis so long positions appear at top,
        # cash at bottom
        ax.set_xlabel("Portfolio Weight", fontsize=10, color=colors["text"])
        ax.set_ylabel("Assets", fontsize=10, color=colors["text"])

        # Set title
        if title is None:
            portfolio_name = self.name if self.name else "Portfolio"
            if self.time_range:
                title = (
                    f"{portfolio_name} Allocation\n{self.time_range[0]} to "
                    f"{self.time_range[1]}"
                )
            else:
                title = f"{portfolio_name} Allocation"

        ax.set_title(title, fontsize=11, pad=15, color=colors["text"])

        # Percentage labels removed for cleaner appearance

        # Set x-axis limits with padding
        max_abs_weight = max(abs(w) for w in all_weights) if all_weights else 0.1
        padding = max_abs_weight * 0.15
        ax.set_xlim(-max_abs_weight - padding, max_abs_weight + padding)

        # Add horizontal separator line between cash and risky assets
        if abs(cash) > cutoff and len(all_tickers) > 1:
            # Find cash position and position separator in the middle of the gap
            cash_ticker_index = next(
                (i for i, ticker in enumerate(all_tickers) if ticker == "CASH"), -1
            )
            if cash_ticker_index >= 0:
                cash_y_pos = y_positions[cash_ticker_index]
                # Find the last risky asset position (just before cash)
                last_risky_y_pos = (
                    y_positions[cash_ticker_index - 1] if cash_ticker_index > 0 else 0
                )
                # Position separator in the middle of the gap
                separator_y = (cash_y_pos + last_risky_y_pos) / 2
                ax.axhline(
                    separator_y,
                    color=colors["text"],
                    linewidth=1,
                    alpha=0.6,
                    linestyle="--",
                )

        # Grid and styling - subtle grid similar to backtest
        ax.grid(True, alpha=0.3, color=colors["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(colors["grid"])
        ax.spines["bottom"].set_color(colors["grid"])

        # Tick styling
        ax.tick_params(axis="both", colors=colors["text"], labelsize=9)
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.tick_params(axis="y", which="both", left=False, right=False)

        # Legend with gradient representation
        legend_elements = []
        if long_positions:
            # Use darkest blue for legend
            num_long = len(long_positions)
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="#1F5F8B",
                    label=f"Long Positions ({num_long})",
                )
            )
        if short_positions:
            # Use darkest red for legend
            num_short = len(short_positions)
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="#8B0000",
                    label=f"Short Positions ({num_short})",
                )
            )
        if abs(cash) > cutoff:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor="#FFD700", label="Cash")
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                fontsize=8,
            )

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_name = f"{self.name.lower()}_allocation.png"
            plt.savefig(
                os.path.join(save_path, save_name),
                dpi=dpi,
                bbox_inches="tight",
                facecolor=colors["background"],
                edgecolor="none",
            )
            print(f"Portfolio plot saved: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()

        return ax

    def save_portfolio(self, save_path):
        """Save portfolio to JSON file."""
        save_weights = self.weights.tolist()

        portfolio = {
            "name": self.name,
            "weights": save_weights,
            "cash": self.cash,
            "tickers": self.tickers,
            "time_range": self.time_range,
        }

        with open(save_path, "w") as json_file:
            json.dump(portfolio, json_file, indent=4)

    def load_portfolio_from_json(self, load_path):
        """Load portfolio from JSON file and update current instance."""

        with open(load_path, "r") as json_file:
            data = json.load(json_file)

        self.name = data["name"]
        self.tickers = data["tickers"]
        self._n_assets = len(self.tickers)
        weights = np.array(data["weights"])
        self._check_self_financing(weights, data["cash"])
        self.weights = weights
        self.cash = data["cash"]

        self.time_range = data["time_range"]
