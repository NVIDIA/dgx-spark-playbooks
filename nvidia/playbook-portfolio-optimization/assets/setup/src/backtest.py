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
Portfolio backtesting and performance evaluation framework.

Provides tools for backtesting portfolio strategies against historical data
and benchmarks, with support for various return metrics and scenario generation
methods including historical, KDE, and Gaussian simulation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

from .portfolio import Portfolio


class portfolio_backtester:
    """
    Portfolio backtesting framework for performance evaluation against benchmarks.

    Supports multiple testing methods including historical data, KDE simulation,
    and Gaussian simulation. Calculates key performance metrics including
    Sharpe ratio, Sortino ratio, and maximum drawdown.

    Attributes
    ----------
    test_portfolio : Portfolio
        Portfolio to be tested
    test_method : str
        Method for generating return scenarios
    benchmark_portfolios : list
        List of benchmark Portfolio objects for comparison
    risk_free_rate : float
        Risk-free rate for excess return calculations
    """

    def __init__(
        self,
        test_portfolio,
        returns_dict,
        risk_free_rate=0.0,
        test_method="historical",
        benchmark_portfolios=None,
    ):
        """
        Initialize portfolio backtester with test portfolio and return data.

        Parameters
        ----------
        test_portfolio : Portfolio
            Portfolio object to backtest
        returns_dict : dict
            Dictionary containing return data with keys: 'return_type', 'returns',
            'dates', 'mean', 'covariance', 'tickers'
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe and Sortino ratio calculations
        test_method : str, default "historical"
            Method for return scenarios: "historical", "kde_simulation",
            or "gaussian_simulation"
        benchmark_portfolios : list, pd.DataFrame, or None, default None
            Benchmark portfolios for comparison. If None, uses equal-weight portfolio
        """

        self.test_portfolio = test_portfolio
        self.test_method = test_method.lower()
        self.benchmark_portfolios = self._generate_benchmark_portfolios(
            benchmark_portfolios
        )

        self._dates = returns_dict["dates"]
        self._return_type = returns_dict["return_type"]
        self._return_mean = returns_dict["mean"]
        self._covariance = returns_dict["covariance"]
        self._returns = returns_dict["returns"]

        self._R = self._get_return_scenarios()

        if self._return_type == "LOG":
            self.risk_free_rate = np.log(1 + risk_free_rate)
        elif self._return_type == "LINEAR":
            self.risk_free_rate = risk_free_rate

        self._backtest_column_names = [
            "returns",
            "cumulative returns",
            "portfolio name",
            "mean portfolio return",
            "sharpe",
            "sortino",
            "max drawdown",
        ]

    def _generate_benchmark_portfolios(self, benchmark_portfolios):
        """
        Generate benchmark portfolios from input specification.

        Parameters
        ----------
        benchmark_portfolios : list, pd.DataFrame, or None
            Benchmark portfolio specification

        Returns
        -------
        list
            List of Portfolio objects to use as benchmarks

        Raises
        ------
        ValueError
            If input format is not supported
        """
        if benchmark_portfolios is None:  # when benchmark_portfolios is not provided,
            # default to equal-weight portfolio
            return self._generate_equal_weights_portfolio(
                self.test_portfolio.tickers, self.test_portfolio.cash
            )
        elif isinstance(
            benchmark_portfolios, pd.DataFrame
        ):  # if custom, then set to input portfolios stored in optimization problem
            return benchmark_portfolios["portfolio"].to_list()
        elif isinstance(benchmark_portfolios, list):
            return benchmark_portfolios
        else:
            raise ValueError(
                "Unacceptable input format.\n Please provide the portfolios "
                "in compliant format (DataFrame)"
            )

    def _generate_equal_weights_portfolio(self, tickers, cash):
        """
        Create equal-weight benchmark portfolio.

        Parameters
        ----------
        tickers : list
            List of asset ticker symbols
        cash : float
            Cash allocation for the portfolio

        Returns
        -------
        list
            List containing single equal-weight Portfolio object
        """
        n_assets = len(tickers)
        weights = (np.ones(n_assets) - cash) / n_assets
        eq_weight_portfolio = Portfolio(
            name="equal-weight", tickers=tickers, weights=weights, cash=cash
        )
        return [eq_weight_portfolio]

    def _generate_simulated_scenarios(self, generation_method="kde", num_scen=5000):
        """
        Generate simulated return scenarios using specified method.

        Parameters
        ----------
        generation_method : str, default "kde"
            Method for scenario generation: "gaussian" or "kde"
        num_scen : int, default 5000
            Number of scenarios to generate

        Returns
        -------
        np.ndarray
            Simulated return scenarios with shape (num_scen, n_assets)

        Raises
        ------
        NotImplementedError
            If generation method is not supported
        """
        generation_method = str(generation_method).lower()
        if generation_method == "gaussian":  # fit Gaussian
            R = np.random.multivariate_normal(
                self._return_mean, self._covariance, size=num_scen
            )

        elif generation_method == "kde":  # kde distribution
            R = self._generate_samples_kde(self._returns, num_scen, bandwidth=0.005)

        else:
            raise NotImplementedError("Invalid Generation Method!")
        return R

    def _generate_samples_kde(
        self, returns_data, num_scen, bandwidth, kernel="gaussian"
    ):
        """
        Generate return samples using Kernel Density Estimation.

        Parameters
        ----------
        returns_data : np.ndarray
            Historical return data for fitting KDE
        num_scen : int
            Number of samples to generate
        bandwidth : float
            Bandwidth parameter for KDE
        kernel : str, default "gaussian"
            Kernel type for density estimation

        Returns
        -------
        np.ndarray
            Generated return samples with shape (num_scen, n_assets)
        """
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(returns_data)
        new_samples = kde.sample(num_scen)

        return new_samples

    def _get_return_scenarios(self):
        """
        Generate return scenarios based on specified test method.

        Returns
        -------
        np.ndarray
            Return scenarios for backtesting with shape (n_scenarios, n_assets)

        Raises
        ------
        NotImplementedError
            If test method is not supported
        """
        if self.test_method == "historical":
            R = self._returns

        elif self.test_method == "kde_simulation":
            R = self._generate_simulated_scenarios(generation_method="kde")

        elif self.test_method == "gaussian_simulation":
            R = self._generate_simulated_scenarios(generation_method="gaussian")

        else:
            raise NotImplementedError("invalid test method!")

        return R

    def backtest_against_benchmarks(
        self,
        plot_returns=False,
        ax=None,
        cut_off_date=None,
        title=None,
        save_plot=False,
        results_dir="results",
    ):
        """
        Backtest portfolio against benchmark portfolios and optionally plot results.

        Parameters
        ----------
        plot_returns : bool, default False
            Whether to create cumulative returns plot
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure
        cut_off_date : str, optional
            Date to mark with vertical line on plot
        title : str, optional
            Title for the plot
        save_plot : bool, default False
            Whether to save the plot to the results directory
        results_dir : str, default "results"
            Directory path where plots will be saved

        Returns
        -------
        tuple
            (backtest_results, ax) where backtest_results is pd.DataFrame
            with performance metrics and ax is the matplotlib axes
        """

        backtest_results = pd.DataFrame({}, columns=self._backtest_column_names)

        # backtest optimal portfolio
        backtest_results = pd.concat(
            [backtest_results, self.backtest_single_portfolio(self.test_portfolio)]
        )

        for portfolio in self.benchmark_portfolios:
            result = self.backtest_single_portfolio(portfolio)
            backtest_results = pd.concat([backtest_results, result], ignore_index=True)

        backtest_results.set_index("portfolio name", inplace=True)
        if plot_returns:
            colors = {
                "frontier": "#88CEE6",
                "benchmark": ["#F6C8A8", "#F18F01", "#C73E1D"],
                "assets": "#7209B7",
                "custom": "#F72585",
                "background": "#FAFAFA",
                "grid": "#E0E0E0",
            }

            # Apply professional styling
            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_context("paper", font_scale=0.9)

            if ax is None:
                fig, ax = plt.subplots(
                    figsize=(12, 8), dpi=300, facecolor=colors["background"]
                )
                ax.set_facecolor(colors["background"])

            # Prepare data for plotting
            cumulative_returns_dataframe = pd.DataFrame(
                [], index=pd.to_datetime(self._dates), columns=backtest_results.index
            )

            for ptf_name, row in backtest_results.iterrows():
                cumulative_returns = row["cumulative returns"]
                cumulative_returns_dataframe[ptf_name] = cumulative_returns

            # Plot each portfolio with consistent colors
            portfolio_names = list(backtest_results.index)
            for i, ptf_name in enumerate(portfolio_names):
                if ptf_name == self.test_portfolio.name:
                    # Test portfolio gets frontier color
                    color = colors["frontier"]
                    linewidth = 2.5
                    alpha = 0.9
                    zorder = 3
                elif "equal-weight" in ptf_name.lower():
                    # Equal-weight benchmark gets same color as buy & hold
                    # in rebalance.py
                    color = colors["benchmark"][1]
                    linewidth = 2
                    alpha = 0.8
                    zorder = 2
                else:
                    # Other benchmarks get rotating benchmark colors
                    color_idx = (i - 1) % len(colors["benchmark"])
                    color = colors["benchmark"][color_idx]
                    linewidth = 1
                    alpha = 0.8
                    zorder = 2

                ax.plot(
                    cumulative_returns_dataframe.index,
                    cumulative_returns_dataframe[ptf_name],
                    linewidth=linewidth,
                    color=color,
                    label=ptf_name,
                    alpha=alpha,
                    zorder=zorder,
                )

            # subtle shading under the test portfolio line
            test_portfolio_data = cumulative_returns_dataframe[self.test_portfolio.name]
            ax.fill_between(
                test_portfolio_data.index,
                test_portfolio_data.values,
                alpha=0.1,
                color=colors["frontier"],
                zorder=1,
            )

            # Dynamically adjust y-axis range to zoom into data with padding
            all_values = []
            for col in cumulative_returns_dataframe.columns:
                all_values.extend(cumulative_returns_dataframe[col].values)

            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min
            padding = y_range * 0.05  # 5% padding on top and bottom

            ax.set_ylim(y_min - padding, y_max + padding)

            # Set labels and formatting
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel(
                f"Cumulative {self._return_type.lower()} returns", fontsize=10
            )
            if title:
                ax.set_title(title, fontsize=11, pad=15)

            # Format legend with smaller font
            ax.legend(
                loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=8
            )

            # Rotate x-axis dates to avoid overlapping
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=9)

            # Set grid style
            ax.grid(True, alpha=0.3, color=colors["grid"])
            ax.set_axisbelow(True)

            if cut_off_date is not None:
                cut_off_date = pd.to_datetime(cut_off_date)
                ax.axvline(
                    x=cut_off_date,
                    color="lightgray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    label="Cut-off Date",
                )

            # Adjust layout to prevent clipping of rotated labels
            plt.tight_layout()

            # Save plot if requested
            if save_plot:
                # Create results directory if it doesn't exist
                os.makedirs(results_dir, exist_ok=True)

                # Generate filename based on test portfolio and date range
                portfolio_name = self.test_portfolio.name.replace(" ", "_").lower()

                # Handle both datetime and string date formats
                if len(self._dates) > 0:
                    try:
                        # Try to use strftime if it's a datetime object
                        start_date = self._dates[0].strftime("%Y%m%d")
                        end_date = self._dates[-1].strftime("%Y%m%d")
                    except AttributeError:
                        # If it's a string, convert to datetime first
                        start_date = pd.to_datetime(self._dates[0]).strftime("%Y%m%d")
                        end_date = pd.to_datetime(self._dates[-1]).strftime("%Y%m%d")
                else:
                    start_date = "unknown"
                    end_date = "unknown"

                test_method = self.test_method.replace("_", "")

                filename = (
                    f"backtest_{portfolio_name}_{test_method}_"
                    f"{start_date}-{end_date}.png"
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

                print(f"Backtest plot saved: {filepath}")

        return backtest_results, ax

    def backtest_single_portfolio(self, portfolio):
        """
        Run backtest for a single portfolio.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio object to backtest

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with backtest results and performance metrics

        Raises
        ------
        NotImplementedError
            If return type is not supported
        """
        if self._return_type == "LOG" or self._return_type == "LINEAR":
            # compute Sharpe Ratio, Sortino Ratio, and Max Drawdown (MDD)
            portfolio_returns = self._compute_portfolio_returns_with_cash(
                portfolio.weights, portfolio.cash
            )
            backtest_result = self._compute_return_metrics(
                portfolio.name, portfolio_returns, portfolio.cash
            )
        else:
            raise NotImplementedError("Return type not supported yet!")

        return backtest_result

    def _compute_return_metrics(self, portfolio_name, returns, cash):
        """
        Calculate portfolio performance metrics.

        Parameters
        ----------
        portfolio_name : str
            Name of the portfolio
        returns : pd.Series or np.ndarray
            Portfolio returns time series
        cash : float
            Cash allocation in the portfolio

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with performance metrics including
            Sharpe ratio, Sortino ratio, and maximum drawdown
        """
        mean_return = np.mean(returns) + cash * self.risk_free_rate
        excess_returns = returns - self.risk_free_rate
        if self._return_type == "LINEAR":
            cumulative_returns = np.cumsum(returns)
        elif self._return_type == "LOG":
            cumulative_returns = np.exp(np.cumsum(returns))

        sharpe = self.sharpe_ratio(excess_returns)
        sortino = self.sortino_ratio(excess_returns)
        mdd = self.max_drawdown(cumulative_returns)

        result = pd.Series(
            [
                returns.to_numpy(),
                cumulative_returns.to_numpy(),
                portfolio_name,
                mean_return,
                sharpe,
                sortino,
                mdd,
            ],
            index=self._backtest_column_names,
        )

        return result.to_frame().T

    def _compute_portfolio_returns_with_cash(self, weights, cash):
        """
        Calculate portfolio returns including cash allocation.

        Parameters
        ----------
        weights : np.ndarray
            Asset weights in the portfolio
        cash : float
            Cash allocation earning risk-free rate

        Returns
        -------
        np.ndarray
            Portfolio returns time series
        """
        return self._R @ weights + self.risk_free_rate * cash

    def sharpe_ratio(self, excess_returns):
        """
        Calculate annualized Sharpe ratio.

        Parameters
        ----------
        excess_returns : np.ndarray
            Excess returns over risk-free rate

        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        mean_excess_return = np.mean(excess_returns)
        std_dev_excess_return = np.std(excess_returns)
        sharpe_ratio = mean_excess_return / std_dev_excess_return * np.sqrt(252)

        return sharpe_ratio

    def sortino_ratio(self, excess_returns):
        """
        Calculate annualized Sortino ratio.

        Parameters
        ----------
        excess_returns : np.ndarray
            Excess returns over risk-free rate

        Returns
        -------
        float
            Annualized Sortino ratio
        """
        mean_excess_return = np.mean(excess_returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns)

        sortino_ratio = mean_excess_return / downside_deviation * np.sqrt(252)

        return sortino_ratio

    def max_drawdown(self, cumulative_returns):
        """
        Calculate maximum drawdown from cumulative returns.

        Parameters
        ----------
        cumulative_returns : np.ndarray
            Cumulative returns time series

        Returns
        -------
        float
            Maximum drawdown as a decimal (e.g., 0.20 for 20% drawdown)
        """
        # Convert log returns to cumulative portfolio values

        # Initial portfolio value (assuming it starts at 1 for simplicity)
        initial_portfolio_value = 1
        portfolio_values = initial_portfolio_value * cumulative_returns

        # Compute the running maximum
        running_max = np.maximum.accumulate(portfolio_values)

        # Compute the max drawdown
        drawdown = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdown)

        return max_drawdown
