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

"""Scenario generation module for portfolio optimization.

Provides tools for generating synthetic financial data using Geometric Brownian Motion
and other stochastic processes. Used to create forward-looking scenarios for
risk assessment and portfolio optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ForwardPathSimulator:
    """Generates synthetic forward paths for financial assets.

    Uses Geometric Brownian Motion to simulate asset price paths based on
    historical data calibration.

    Parameters
    ----------
    fitting_data : pd.DataFrame
        Historical price data for calibration (dates x assets).
    generation_dates : pd.DatetimeIndex or list
        Date range for synthetic data generation.
    n_paths : int
        Number of scenarios/forward paths to generate.
    method : str, default "log_gbm"
        Generation method (currently only "log_gbm" supported).

    Attributes
    ----------
    fitting_data : pd.DataFrame
        Historical data used for calibration.
    dates : pd.DatetimeIndex or list
        Generation date range.
    n_steps : int
        Number of time steps for simulation.
    n_paths : int
        Number of scenarios generated.
    generation_method : str
        Method used for generation.
    simulated_paths : np.ndarray
        Generated synthetic paths (n_paths x n_steps+1 x n_assets).
    """

    def __init__(self, fitting_data, generation_dates, n_paths, method="log_gbm"):
        """Initialize scenario generator with data and parameters."""
        self.fitting_data = fitting_data
        self.dates = generation_dates
        self.n_steps = len(generation_dates) - 1
        self.n_paths = n_paths
        self.generation_method = method.lower()

    def generate(self, plot_paths=False, n_plots=0):
        """Generate synthetic forward paths.

        Parameters
        ----------
        plot_paths : bool, default False
            Whether to plot generated paths.
        n_plots : int, default 0
            Number of paths to plot if plot_paths is True.

        Raises
        ------
        ValueError
            If generation method is not recognized.
        """
        if self.generation_method == "log_gbm":
            mu, sigma, L = self._calibrate_log_process()
            self.simulated_paths = self._generate_via_log_gbm(mu, sigma, L)
        else:
            raise ValueError("Unrecognized generation method.")

        if plot_paths:
            self._plot_generated_paths(n_plots)

    def _calibrate_log_process(self):
        """Calibrate log-normal process parameters from historical data.

        Returns
        -------
        mu : np.ndarray
            Drift parameters for each asset.
        sigma : np.ndarray
            Covariance matrix of log returns.
        L : np.ndarray
            Cholesky decomposition of covariance matrix.
        """
        log_returns = np.log(self.fitting_data / self.fitting_data.shift(1)).dropna()

        # Estimate covariance matrix of log returns
        sigma = log_returns.cov().values
        # Cholesky decomposition of the correlation matrix
        L = np.linalg.cholesky(sigma).T

        # Estimate drift
        total_drift = log_returns.iloc[-1].values - log_returns.iloc[0].values
        step_drift = total_drift / self.n_steps
        mu = step_drift + 0.5 * np.sum(L**2, axis=1)

        return mu, sigma, L

    def _generate_via_log_gbm(self, mu, sigma, L, dt=1):
        """Generate paths using log-normal Geometric Brownian Motion.

        Parameters
        ----------
        mu : np.ndarray
            Drift parameters for each asset.
        sigma : np.ndarray
            Covariance matrix of log returns.
        L : np.ndarray
            Cholesky decomposition of covariance matrix.
        dt : float, default 1
            Time step size.

        Returns
        -------
        np.ndarray
            Simulated paths (n_paths x n_steps+1 x n_assets).
        """
        # Initial forward rates

        last_rates = self.fitting_data.loc[
            self.dates[0]
        ].values  # set starting value as the start of the generation period

        # Initialize an array for simulated paths
        simulated_paths = np.zeros((self.n_paths, self.n_steps + 1, len(mu)))

        current_rates = last_rates
        simulated_paths[:, 0, :] = current_rates
        Z = np.random.normal(size=(self.n_paths, self.n_steps, len(mu)))
        dW = np.matmul(Z, L) * np.sqrt(dt)

        for t in range(1, self.n_steps + 1):
            # compute drift and diffusion
            drift = (mu - 0.5 * np.diag(sigma) ** 2) * dt
            diffusion = dW[:, t - 1, :]

            # Simulate next step forward rates using GBM formula
            simulated_paths[:, t, :] = simulated_paths[:, t - 1, :] * np.exp(
                drift + diffusion
            )

        return simulated_paths

    def _plot_generated_paths(self, n_plots):
        """Plot randomly selected generated paths.

        Parameters
        ----------
        n_plots : int
            Number of paths to plot.
        """
        # Assuming 'simulated_paths' is your array of simulated paths with shape
        # (n_paths, n_steps, n_ccy_pairs)
        n_paths = self.simulated_paths.shape[0]
        _ = self.simulated_paths.shape[2]  # n_ccy_pairs (unused)

        # Randomly select indices for the scenarios to plot
        random_indices = np.random.choice(n_paths, n_plots, replace=False)
        plt.rcParams.update({"font.size": 8})
        sns.set(rc={"figure.dpi": 100, "savefig.dpi": 300})
        sns.set_palette(palette="tab10")
        sns.set_style("white")

        # Loop through each selected scenario and create a subplot
        for i, idx in enumerate(random_indices):
            plt.figure(i, figsize=(10, 7))

            selected_paths = pd.DataFrame(
                self.simulated_paths[idx, :, :],
                index=self.fitting_data.index,
                columns=self.fitting_data.columns,
            )

            selected_paths.plot()

            plt.title(f"Scenario {i + 1} - Path {idx + 1}")
            plt.xticks(rotation=50, fontsize=8)

            plt.ylabel("Forward Rate")
            plt.legend()

            plt.show()

    def get_simulated_paths_ccy_pair(self, ccy_pair):
        """Extract simulated paths for a specific asset.

        Parameters
        ----------
        ccy_pair : str
            Asset identifier to extract paths for.

        Returns
        -------
        pd.DataFrame
            Simulated paths for the specified asset (dates x n_paths).
        """
        ccy_pair_idx = list(self.fitting_data.columns).index(ccy_pair)
        simulated_paths_ccy_pair = self.simulated_paths[:, :, ccy_pair_idx]
        simulated_paths_dataframe = pd.DataFrame(
            simulated_paths_ccy_pair, index=self.dates
        )

        return simulated_paths_dataframe


def generate_synthetic_stock_data(
    dataset_directory, num_synthetic, fit_range, generate_range
):
    """Generate synthetic stock data using Geometric Brownian Motion.

    Fits GBM parameters to historical data from one period and generates
    synthetic time series for another period.

    Parameters
    ----------
    dataset_directory : str
        Path to CSV file containing historical stock data.
    num_synthetic : int
        Multiplier for synthetic stocks. Total synthetic stocks will be
        num_synthetic * num_assets.
    fit_range : tuple of str
        Start and end dates for calibration period (start, end).
    generate_range : tuple of str
        Start and end dates for generation period (start, end).

    Returns
    -------
    pd.DataFrame
        Combined dataset with original and synthetic stock data.
        Synthetic columns are named as 'ticker-idx' where idx is the
        path number.
    """
    input_data = pd.read_csv(dataset_directory, index_col=0)
    fit_data = input_data.loc[fit_range[0] : fit_range[1]]
    n_assets = len(fit_data.columns)
    generate_time_range = input_data.loc[generate_range[0] : generate_range[1]].index

    scen_gen = ForwardPathSimulator(
        fitting_data=fit_data,
        generation_dates=generate_time_range,
        n_paths=num_synthetic,
        method="log_gbm",
    )

    scen_gen.generate()

    synthetic_data = scen_gen.simulated_paths.transpose(1, 0, 2).reshape(
        scen_gen.n_steps + 1, (scen_gen.n_paths * n_assets)
    )

    tickers_list = list(input_data.columns)

    synthetic_dataframe = pd.DataFrame(synthetic_data, index=generate_time_range)

    augmented_data = pd.concat(
        [input_data.loc[generate_range[0] : generate_range[1]], synthetic_dataframe],
        axis=1,
    )
    columns = [
        ticker + "-" + str(idx)
        for idx in range(scen_gen.n_paths)
        for ticker in tickers_list
    ]
    tickers_list += columns
    augmented_data.columns = tickers_list

    return augmented_data
