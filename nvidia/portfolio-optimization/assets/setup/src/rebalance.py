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

"""
Dynamic portfolio rebalancing with CVaR optimization.

Implements systematic rebalancing strategies using Conditional Value-at-Risk
optimization with configurable trigger conditions based on portfolio drift,
performance thresholds, or maximum drawdown.

Key Features
------------
* Dynamic rebalancing with multiple trigger conditions
* Rolling CVaR optimization over trading period
* Transaction cost modeling
* Performance visualization and baseline comparison
* Support for various rebalancing criteria

Classes
-------
rebalance_portfolio
    Main class for implementing dynamic rebalancing strategies with CVaR
    optimization and configurable trigger conditions.

"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from . import backtest, cvar_optimizer, cvar_parameters, cvar_utils, portfolio, utils


class rebalance_portfolio:
    """
    Dynamic portfolio rebalancing with CVaR optimization.

    Performs rolling CVaR optimization over a trading period, triggering portfolio
    rebalancing when specified criteria are met (drift, percentage change, or
    maximum drawdown).

    Parameters
    ----------
    dataset_directory : str
        Path to asset universe data.
    returns_compute_settings : dict
        Settings for computing returns.
    scenario_generation_settings : dict
        Settings for generating return scenarios.
    trading_start, trading_end : str
        Trading period boundaries in YYYY-MM-DD format.
    look_forward_window : int
        Backtest evaluation window size in trading days.
    look_back_window : int
        Historical data window for optimization in trading days.
    cvar_params : CvarParameters
        CVaR optimization parameters and constraints.
    solver_settings : dict
        Solver configuration for optimization backend.
    re_optimize_criteria : dict
        Rebalancing trigger conditions with type and threshold.
    print_opt_result : bool, default False
        Whether to print detailed optimization results.
    """

    def __init__(
        self,
        dataset_directory: str,
        returns_compute_settings: dict,
        scenario_generation_settings: dict,
        trading_start: str,
        trading_end: str,
        look_forward_window: int,
        look_back_window: int,
        cvar_params: cvar_parameters.CvarParameters,
        solver_settings: dict,
        re_optimize_criteria: dict,
        print_opt_result: bool = False,
    ):
        """Initialize rebalancing portfolio with optimization parameters."""
        self.dataset_directory = dataset_directory
        self.trading_start = pd.to_datetime(trading_start)
        self.trading_end = pd.to_datetime(trading_end)

        self.look_forward_window = look_forward_window
        self.look_back_window = look_back_window

        self.cvar_params = cvar_params
        self.solver_settings = solver_settings
        self.returns_compute_settings = returns_compute_settings
        self.scenario_generation_settings = scenario_generation_settings
        self.print_opt_result = print_opt_result

        self.re_optimize_criteria = re_optimize_criteria
        self.re_optimize_type = re_optimize_criteria["type"].lower()
        self.re_optimize_threshold = re_optimize_criteria["threshold"]

        self._get_price_data()

        self.dates_range = self.price_data.loc[trading_start:trading_end].index

        (
            self.buy_and_hold_results,
            self.buy_and_hold_cumulative_portfolio_value,
        ) = self._get_buy_and_hold_results()

    def _get_price_data(self):
        """Load price data and calculate returns based on specified return type."""
        self.price_data = pd.read_csv(
            self.dataset_directory, index_col=0, parse_dates=True
        )
        price_data_start = self.trading_start - pd.Timedelta(days=self.look_back_window)
        assert (
            price_data_start >= self.price_data.index[0]
        ), "Invalid start date - choose a later date!"

    def re_optimize(
        self,
        transaction_cost_factor: float = 0,
        plot_results: bool = False,
        existing_portfolio: portfolio.Portfolio = None,
        run_re_optimize: bool = True,
        save_plot: bool = False,
        results_dir: str = "results",
    ):
        """Execute rebalancing strategy over the trading period.

        Performs rolling optimization and backtesting, triggering rebalancing
        when specified criteria are met.

        Parameters
        ----------
        transaction_cost_factor : float, default 0
            Transaction cost as fraction of turnover.
        plot_results : bool, default False
            Whether to generate performance plots.
        existing_portfolio : Portfolio, optional
            Starting portfolio allocation.
        run_re_optimize : bool, default True
            Enable rebalancing logic. If False, uses initial portfolio only.
        save_plot : bool, default False
            Whether to save generated plots.
        results_dir : str, default "results"
            Directory for saving plots.

        Returns
        -------
        results_dataframe : pd.DataFrame
            Detailed results for each rebalancing decision.
        re_optimize_dates : list
            Dates when rebalancing was triggered.
        cumulative_portfolio_value : pd.Series
            Time series of portfolio values.
        """
        if run_re_optimize:
            print(f"{'=' * 60}")
            print("DYNAMIC REBALANCING ANALYSIS")
            print(
                f"Period: {self.trading_start.strftime('%Y-%m-%d')} to "
                f"{self.trading_end.strftime('%Y-%m-%d')}"
            )
            # Format strategy name with special handling for pct_change
            strategy_name = self.re_optimize_type.replace("_", " ").title()
            if strategy_name == "Pct Change":
                strategy_name = "Percentage Change"
            print(f"Strategy: {strategy_name}")
            print(f"Threshold: {self.re_optimize_threshold}")
            print(f"Look-forward window: {self.look_forward_window} days")
            print(f"Look-back window: {self.look_back_window} days")
            print(f"{'=' * 60}")
        result = 0.0
        re_optimize_flag = False
        portfolio_value = 1.0

        current_portfolio = self.initial_portfolio if run_re_optimize else None

        results_dataframe = pd.DataFrame(
            columns=[
                self.re_optimize_type,
                "re_optimized",
                "portfolio_value",
                "solver_time",
            ]
        )
        _ = pd.DataFrame(columns=["portfolio_value"])  # no_re_optimize_results (unused)

        cumulative_portfolio_value_array = np.array([])
        cumulative_portfolio_value_dates = []  # Track dates for portfolio values
        re_optimize_dates = []

        backtest_idx = 0
        backtest_date = self.trading_start
        backtest_final_date = self.dates_range[-self.look_forward_window]

        while backtest_date < backtest_final_date:
            results_dataframe.loc[backtest_date, self.re_optimize_type] = result
            results_dataframe.loc[backtest_date, "re_optimized"] = re_optimize_flag
            results_dataframe.loc[backtest_date, "optimal_portfolio"] = (
                current_portfolio
            )
            results_dataframe.loc[backtest_date, "portfolio_value"] = portfolio_value
            # Use initial solve time for the first row when rebalancing
            # (shared from buy-and-hold)
            if backtest_date == self.trading_start and run_re_optimize:
                results_dataframe.loc[backtest_date, "solver_time"] = (
                    self.buy_and_hold_results.iloc[0]["solver_time"]
                )
            else:
                results_dataframe.loc[backtest_date, "solver_time"] = None

            existing_portfolio = current_portfolio

            # re-optimize if criteria goes beyond threshold
            if (
                (backtest_date == self.trading_start) and not run_re_optimize
            ) or re_optimize_flag:
                optimize_start = backtest_date - pd.Timedelta(
                    days=self.look_back_window
                )
                optimize_regime = {
                    "name": "re-optimize",
                    "range": (
                        optimize_start.strftime("%Y-%m-%d"),
                        backtest_date.strftime("%Y-%m-%d"),
                    ),
                }

                optimize_returns_dict = utils.calculate_returns(
                    self.price_data,
                    optimize_regime,
                    self.returns_compute_settings
                )
                optimize_returns_dict = cvar_utils.generate_cvar_data(
                    optimize_returns_dict, 
                    self.scenario_generation_settings
                )

                re_optimize_problem = cvar_optimizer.CVaR(
                    returns_dict=optimize_returns_dict,
                    cvar_params=self.cvar_params,
                    existing_portfolio=existing_portfolio,
                )

                result_row, current_portfolio = (
                    re_optimize_problem.solve_optimization_problem(
                        self.solver_settings,
                        print_results=self.print_opt_result,
                    )
                )
                if (backtest_date == self.trading_start) and not run_re_optimize:
                    self.initial_portfolio = current_portfolio

                # Store solver time information
                solve_time = result_row.get("solve time", None)
                results_dataframe.at[backtest_date, "solver_time"] = solve_time

                re_optimize_dates.append(backtest_date)

                if run_re_optimize:
                    print(
                        f"Rebalancing triggered on "
                        f"{backtest_date.strftime('%Y-%m-%d')} | "
                        f"Event #{len(re_optimize_dates)} | "
                        f"Portfolio value: ${portfolio_value:,.2f}"
                    )

            # get backtest start and end dates
            backtest_start = backtest_date.strftime("%Y-%m-%d")
            backtest_end = self.dates_range[backtest_idx + self.look_forward_window]
            backtest_end = backtest_end.strftime("%Y-%m-%d")
            backtest_regime = {
                "name": "backtest",
                "range": (backtest_start, backtest_end),
            }

            # calculate returns
            test_returns_dict = utils.calculate_returns(
                self.price_data, backtest_regime, self.returns_compute_settings
            )

            # run backtest
            backtester = backtest.portfolio_backtester(
                current_portfolio, test_returns_dict, benchmark_portfolios=None
            )
            backtest_result = backtester.backtest_single_portfolio(current_portfolio)
            cur_cumulative_portfolio_returns = (
                backtest_result["cumulative returns"].values[0] * portfolio_value
            )

            cumulative_portfolio_value_array = np.concatenate(
                (cumulative_portfolio_value_array, cur_cumulative_portfolio_returns)
            )

            # Extract actual trading dates from the backtester object
            backtest_period_dates = backtester._dates
            cumulative_portfolio_value_dates.extend(backtest_period_dates)

            # update portfolio value
            portfolio_value_pct_change = self._calculate_pct_change(backtest_result)
            transaction_cost = self._calculate_transaction_cost(
                current_portfolio, existing_portfolio, transaction_cost_factor
            )
            portfolio_value = portfolio_value * (
                1 + portfolio_value_pct_change - transaction_cost
            )

            # re-optimize criteria check
            if run_re_optimize:
                if self.re_optimize_type == "pct_change":
                    result, re_optimize_flag = self._check_pct_change(
                        portfolio_value_pct_change, backtest_result, results_dataframe
                    )
                elif self.re_optimize_type == "drift_from_optimal":
                    # Calculate the index for drift checking
                    result, re_optimize_flag = self._check_drift_from_optimal(
                        current_portfolio, backtest_idx
                    )

                elif self.re_optimize_type == "max_drawdown":
                    result = backtest_result["max drawdown"].values[0]
                    re_optimize_flag = self._check_max_drawdown(result)

                elif self.re_optimize_type == "no_re_optimize":
                    re_optimize_flag = False
                    result = None

            backtest_idx += self.look_forward_window
            backtest_date = self.dates_range[backtest_idx]

        # Convert to pandas Series with dates as index, ensuring proper datetime format
        cumulative_portfolio_value_dates_clean = pd.to_datetime(
            cumulative_portfolio_value_dates
        )
        cumulative_portfolio_value = pd.Series(
            cumulative_portfolio_value_array,
            index=cumulative_portfolio_value_dates_clean,
            name="cumulative_portfolio_value",
        )

        # Print analysis summary
        if run_re_optimize:
            total_return = (
                cumulative_portfolio_value.iloc[-1] / cumulative_portfolio_value.iloc[0]
                - 1
            ) * 100
            print("\nANALYSIS COMPLETE")
            print(f"Total rebalancing events: {len(re_optimize_dates)}")
            print(f"Final portfolio value: ${cumulative_portfolio_value.iloc[-1]:,.2f}")
            print(f"Total return: {total_return:+.2f}%")
            print(f"Data points collected: {len(cumulative_portfolio_value):,}")
            print(f"{'=' * 60}\n")

        if plot_results:
            self.plot_results(
                results_dataframe,
                re_optimize_dates,
                cumulative_portfolio_value,
                save_plot,
                results_dir,
            )

        return results_dataframe, re_optimize_dates, cumulative_portfolio_value

    def _calculate_pct_change(self, backtest_result: pd.DataFrame):
        """Calculate percentage change in portfolio value over backtest period.

        Parameters
        ----------
        backtest_result : pd.DataFrame
            Backtest results containing cumulative returns.

        Returns
        -------
        float
            Percentage change from start to end of period.
        """
        cumulative_returns = backtest_result["cumulative returns"][0]
        pct_change = (
            cumulative_returns[-1] / cumulative_returns[0] - 1
        )  # percent change

        return pct_change

    def _calculate_transaction_cost(
        self,
        current_portfolio: portfolio.Portfolio,
        existing_portfolio: portfolio.Portfolio,
        transaction_cost_factor: float,
    ):
        """Calculate transaction costs from portfolio rebalancing.

        Parameters
        ----------
        current_portfolio : Portfolio
            New target allocation.
        existing_portfolio : Portfolio
            Previous allocation.
        transaction_cost_factor : float
            Cost factor as fraction of turnover.

        Returns
        -------
        float
            Total transaction cost.
        """
        if existing_portfolio is None:
            return 0
        else:
            turnover = np.sum(
                np.abs(current_portfolio.weights - existing_portfolio.weights)
            )
            return turnover * transaction_cost_factor

    def _check_pct_change(
        self,
        pct_change: float,
        backtest_result: pd.DataFrame,
        results_dataframe: pd.DataFrame,
    ):
        """Check if percentage change triggers rebalancing.

        Evaluates current and cumulative negative returns against threshold.

        Parameters
        ----------
        pct_change : float
            Current period percentage change.
        backtest_result : pd.DataFrame
            Current backtest results.
        results_dataframe : pd.DataFrame
            Historical results for cumulative calculation.

        Returns
        -------
        pct_change : float
            Input percentage change.
        re_optimize_flag : bool
            True if rebalancing should be triggered.
        """
        re_optimize_flag = False

        prev_total = 0
        for idx in reversed(range(results_dataframe.shape[0])):
            if (
                results_dataframe["pct_change"].iloc[idx] < 0
                and not results_dataframe["re_optimized"].iloc[idx]
            ):
                prev_total += results_dataframe["pct_change"].iloc[idx]
            else:
                break

        if (
            pct_change < self.re_optimize_threshold
            or prev_total + pct_change < self.re_optimize_threshold
        ):
            re_optimize_flag = True

        return pct_change, re_optimize_flag

    def _check_drift_from_optimal(
        self, optimal_portfolio: portfolio.Portfolio, backtest_idx: int
    ):
        """Check if portfolio drift from optimal triggers rebalancing.

        Calculates deviation between current and optimal allocation after
        price movements.

        Parameters
        ----------
        optimal_portfolio : Portfolio
            Target optimal allocation.
        backtest_idx : int
            Current backtest date index.

        Returns
        -------
        deviation : float
            Drift magnitude (L1 or L2 norm).
        re_optimize_flag : bool
            True if drift exceeds threshold.
        """

        re_optimize_flag = False

        price_change = self.price_data.iloc[backtest_idx].div(
            self.price_data.iloc[backtest_idx + self.look_forward_window]
        )

        cur_portfolio_weights = price_change.to_numpy() * optimal_portfolio.weights

        if self.re_optimize_criteria["norm"] == 2:
            deviation = np.sum(
                np.abs(cur_portfolio_weights - optimal_portfolio.weights) ** 2
            )  # squared differences
        elif self.re_optimize_criteria["norm"] == 1:
            deviation = np.sum(
                np.abs(cur_portfolio_weights - optimal_portfolio.weights)
            )  # 1-norm

        if deviation > self.re_optimize_threshold:
            re_optimize_flag = True

        return deviation, re_optimize_flag

    def _check_max_drawdown(self, mdd: float):
        """Check if maximum drawdown triggers rebalancing.

        Parameters
        ----------
        mdd : float
            Current maximum drawdown.

        Returns
        -------
        bool
            True if drawdown exceeds threshold.
        """
        re_optimize_flag = False

        if mdd > self.re_optimize_threshold:
            re_optimize_flag = True

        return re_optimize_flag

    def plot_results(
        self,
        results_dataframe: pd.DataFrame,
        re_optimize_dates: list,
        cumulative_portfolio_value: pd.Series,
        save_plot: bool = False,
        results_dir: str = "results",
    ):
        """Generate portfolio performance comparison plots.

        Compares dynamic rebalancing strategy against buy-and-hold baseline
        with rebalancing event markers.

        Parameters
        ----------
        results_dataframe : pd.DataFrame
            Rebalancing results and metrics.
        re_optimize_dates : list
            Dates when rebalancing was triggered.
        cumulative_portfolio_value : pd.Series
            Time series of portfolio values.
        save_plot : bool, default False
            Whether to save the plot.
        results_dir : str, default "results"
            Directory for saving plots.
        """

        # Use the same styling as efficient frontier
        color_schemes = {
            "modern": {
                "frontier": "#7cd7fe",
                "benchmark": ["#ef9100", "#ff8181", "#0d8473"], #NVIDIA orange, red, dark teal
                "assets": "#c359ef",
                "custom": "#fc79ca",
                "background": "#FFFFFF",
                "grid": "#E0E0E0",
            }
        }
        colors = color_schemes["modern"]

        # Create figure with same styling as efficient frontier
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=1.6)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300, facecolor=colors["background"])
        ax.set_facecolor(colors["background"])

        # Plot rebalancing strategy line with proper datetime handling
        ax.plot(
            cumulative_portfolio_value.index,
            cumulative_portfolio_value.values,
            linewidth=3,
            color=colors["frontier"],
            label="Dynamic Rebalancing",
            zorder=3,
            alpha=0.9,
        )

        # Plot no-rebalancing baseline with proper datetime handling
        ax.plot(
            self.buy_and_hold_cumulative_portfolio_value.index,
            self.buy_and_hold_cumulative_portfolio_value.values,
            linewidth=2.5,
            color=colors["benchmark"][0],
            linestyle="-",
            label="Buy & Hold",
            zorder=2,
            alpha=0.8,
        )

        # Add subtle fill under the rebalancing line
        ax.fill_between(
            cumulative_portfolio_value.index,
            cumulative_portfolio_value.values,
            alpha=0.1,
            color=colors["frontier"],
            zorder=1,
        )

        # Add rebalancing date markers as scatter points
        if re_optimize_dates:
            rebalancing_values = []
            rebalancing_dates_clean = []

            for date in re_optimize_dates:
                # Convert date to pandas timestamp if needed
                date_ts = pd.to_datetime(date)

                # Find the portfolio value at this rebalancing date
                if date_ts in cumulative_portfolio_value.index:
                    rebalancing_values.append(cumulative_portfolio_value[date_ts])
                    rebalancing_dates_clean.append(date_ts)
                else:
                    # Find nearest date
                    nearest_idx = cumulative_portfolio_value.index.get_indexer(
                        [date_ts], method="nearest"
                    )[0]
                    if nearest_idx >= 0:
                        nearest_date = cumulative_portfolio_value.index[nearest_idx]
                        rebalancing_values.append(
                            cumulative_portfolio_value[nearest_date]
                        )
                        rebalancing_dates_clean.append(nearest_date)

            if rebalancing_dates_clean:
                # Calculate y-axis range for consistent line heights
                y_min = cumulative_portfolio_value.min()
                y_max = cumulative_portfolio_value.max()
                y_range = y_max - y_min
                line_height = y_range * 0.05  # 5% of y-axis range

                # Create small vertical line segments at each rebalancing date
                for date, value in zip(rebalancing_dates_clean, rebalancing_values):
                    ax.vlines(
                        date,
                        value - line_height,
                        value + line_height,
                        color=colors[
                            "assets"
                        ],  # Use purple color for better visibility
                        linewidth=2.5,
                        alpha=0.9,
                        linestyle="--",  # Dashed lines for visibility
                        zorder=5,
                    )
                    ax.plot(
                        date,
                        value,
                        "o",
                        color=colors["assets"],
                        markersize=7,
                        markeredgecolor="white",
                        markeredgewidth=1.5,
                        zorder=6,
                    )

        # Professional styling
        ax.set_xlabel("Date", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Portfolio Value", fontsize=14, fontweight="bold")

        # Create title based on rebalancing criteria
        rebalance_type = self.re_optimize_type.replace("_", " ").title()
        if rebalance_type == "Pct Change":
            rebalance_type = "Percentage Change"
        title = f"\n{rebalance_type} Rebalancing Strategy"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Grid and styling
        ax.grid(True, alpha=0.3, color=colors["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

        # Clear automatic pandas legend and create custom legend
        if ax.legend_:
            ax.legend_.remove()

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=colors["frontier"],
                linewidth=3,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="Dynamic Rebalancing",
            ),
            Line2D(
                [0],
                [0],
                color=colors["benchmark"][0],
                linewidth=2.5,
                label="Buy & Hold",
            ),
        ]

        # Add rebalancing dates to legend only if they exist
        if re_optimize_dates:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors["assets"],
                    linewidth=2.5,
                    linestyle="--",
                    marker="o",
                    markersize=7,
                    markerfacecolor=colors["assets"],
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    label="Rebalancing Dates",
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            fontsize=12,
        )

        # Format x-axis for better date display
        import matplotlib.dates as mdates

        # Ensure the x-axis is treated as datetime
        ax.xaxis_date()

        # Set reasonable date formatting based on date range
        date_range = (
            cumulative_portfolio_value.index.max()
            - cumulative_portfolio_value.index.min()
        )
        if date_range.days > 365:
            # For longer periods, show year-month
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        else:
            # For shorter periods, show month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", labelrotation=45)

        # Set x-axis limits to the actual data range
        ax.set_xlim(
            cumulative_portfolio_value.index.min(),
            cumulative_portfolio_value.index.max(),
        )

        # Set y-axis limits to zoom into the actual data range with some padding
        all_values = []
        all_values.extend(cumulative_portfolio_value.values)
        all_values.extend(self.buy_and_hold_cumulative_portfolio_value.values)

        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        padding = y_range * 0.05  # 5% padding on top and bottom

        ax.set_ylim(y_min - padding, y_max + padding)

        # Tight layout
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)

            # Generate descriptive filename
            _ = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp (unused)
            strategy_name = self.re_optimize_type.replace("_", "-")
            start_date = self.trading_start.strftime("%Y%m%d")
            end_date = self.trading_end.strftime("%Y%m%d")
            _ = len(re_optimize_dates)  # num_rebalances (unused)

            filename = (
                f"rebalancing_{strategy_name}_with_threshold_"
                f"{self.re_optimize_threshold}_{start_date}-{end_date}events.png"
            )
            filepath = os.path.join(results_dir, filename)

            # Save with high quality
            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

            print(f"Plot saved: {filepath}")

    def _get_buy_and_hold_results(self):
        """Generate baseline buy-and-hold results for comparison.

        Optimizes portfolio once at start and maintains allocation throughout
        trading period.

        Returns
        -------
        buy_and_hold_results_dataframe : pd.DataFrame
            Baseline strategy results.
        cumulative_portfolio_value : pd.Series
            Time series of baseline portfolio values.
        """
        print("=" * 60)
        print("BASELINE (BUY & HOLD) ANALYSIS")
        print(
            f"Period: {self.trading_start.strftime('%Y-%m-%d')} to "
            f"{self.trading_end.strftime('%Y-%m-%d')}"
        )
        print("Strategy: Single optimization at start")
        print("=" * 60)
        (
            buy_and_hold_results_dataframe,
            _,
            cumulative_portfolio_value,
        ) = self.re_optimize(
            plot_results=False, existing_portfolio=None, run_re_optimize=False
        )

        # Print baseline summary
        total_return = (
            cumulative_portfolio_value.iloc[-1] / cumulative_portfolio_value.iloc[0] - 1
        ) * 100
        print("\nBASELINE COMPLETE")
        print(f"Final portfolio value: ${cumulative_portfolio_value.iloc[-1]:,.2f}")
        print(f"Total return: {total_return:+.2f}%")
        print(f"Data points collected: {len(cumulative_portfolio_value):,}")
        print(f"{'=' * 60}\n")

        return buy_and_hold_results_dataframe, cumulative_portfolio_value

    def plot_weights_vs_prices(self, re_optimize_results: pd.DataFrame, ticker: str):
        """Plot portfolio weights evolution against price movements.

        Creates dual-axis plot showing asset prices and portfolio weight
        allocations over time.

        Parameters
        ----------
        re_optimize_results : pd.DataFrame
            Rebalancing results with optimal portfolios.
        ticker : str
            Asset ticker symbol. Must exist in asset universe.

        Raises
        ------
        AssertionError
            If ticker not found in price data.
        """
        assert (
            ticker in self.price_data.columns
        ), "The selected ticker is not in the asset universe!"

        ticker_idx = list(self.price_data.columns).index(ticker)
        ticker_weights_history = [
            current_portfolio.weights[ticker_idx]
            for current_portfolio in re_optimize_results["optimal_portfolio"].iloc[1:]
        ]
        fig, ax1 = plt.subplots(figsize=(12, 6))
        plot_start_date = re_optimize_results.index[1]
        plot_end_date = re_optimize_results.index[-1]
        price_data = self.price_data.loc[plot_start_date:plot_end_date, ticker]
        ax1.plot(price_data, color="red", label=f"{ticker} prices")
        ax1.set_title(f"{ticker} weights vs. prices")

        ax2 = ax1.twinx()
        ax2.bar(
            re_optimize_results.index[1:],
            ticker_weights_history,
            color="#76b900",
            width=30,
            label=f"{ticker}",
            alpha=0.7,
        )
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax1.legend()
        ax2.legend()
        # Show the plot
        plt.tight_layout()
        plt.show()
