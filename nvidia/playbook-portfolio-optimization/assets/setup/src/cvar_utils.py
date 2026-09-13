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

import os
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.neighbors

from . import scenario_generation, utils
from .cvar_data import CvarData
from .cvar_parameters import CvarParameters
from .portfolio import Portfolio

# Note: cvar_optimizer and cuml are imported lazily within functions to avoid
# circular imports and loading CUDA libraries at module import time

def generate_samples_kde(
    num_scen: int,
    returns_data: np.ndarray,
    kde_settings: dict = None,
    verbose: bool = False,
):
    """Fit KernelDensity to data and return new samples.

    Args:
        num_scen (int): Number of scenarios to generate.
        returns_data (np.ndarray): Historical returns data for fitting.
        kde_settings (dict, optional): Dictionary containing KDE settings. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Array of generated samples with shape (num_scen, n_features).

    Raises:
        ValueError: If device is not "CPU" or "GPU".

    Example:
        >>> import numpy as np
        >>> # Historical returns for 3 assets over 100 days
        >>> returns_data = np.random.randn(100, 3) * 0.02
        >>> # Generate 50 new scenarios using KDE
        >>> new_scenarios = generate_samples_kde(
        ...     num_scen=50,
        ...     returns_data=returns_data,
        ...     bandwidth=0.01,
        ...     kernel="gaussian",
        ...     device="CPU"
        ... )
        >>> print(new_scenarios.shape)  # (50, 3)
    """
    kde_device = kde_settings["device"]
    bandwidth = kde_settings["bandwidth"]
    kernel = kde_settings["kernel"]

    if kde_device == "CPU":
        kde = sklearn.neighbors.KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
            returns_data
        )
        new_samples = kde.sample(num_scen)
        if verbose:
            print("KDE fitting on CPU")

    elif kde_device == "GPU":
        # Lazy import to avoid loading CUDA libraries on module import
        import cuml.neighbors

        kde = cuml.neighbors.KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
            returns_data
        )
        new_samples = kde.sample(num_scen).get()  # convert to numpy array
        if verbose:
            print("KDE fitting on GPU")

    else:
        raise ValueError("Invalid Device: CPU or GPU!")

    return new_samples


def generate_cvar_data(returns_dict: dict, scenario_generation_settings: dict):
    """Generate CvarData dataclass for CVaR optimization.

    This function creates the CvarData dataclass containing scenarios and probabilities
    based on the specified fit type (Gaussian, KDE, or historical).

    Args:
        returns_dict (dict): Dictionary containing returns data with mean,
            covariance, and returns.
        scenario_generation_settings (dict): Dictionary containing scenario generation settings including
            fit_type and num_scen.

    Returns:
        dict: Updated returns_dict with added 'cvar_data' containing
            CvarData dataclass.

    Raises:
        ValueError: If fit_type is not "gaussian", "kde", or "no_fit".

    Example:
        >>> import numpy as np
        >>> # Prepare returns data
        >>> returns_dict = {
        ...     "mean": np.array([0.08, 0.10, 0.12]),
        ...     "covariance": np.eye(3) * 0.01,
        ...     "returns": np.random.randn(100, 3) * 0.02
        ... }
        >>> scenario_generation_settings = {"num_scen": 50, "fit_type": "gaussian"}
        >>> result = generate_CVaR_data(returns_dict, scenario_generation_settings)
        >>> print(type(result["cvar_data"]))
        <class 'cufolio.cvar_data.CvarData'>
        >>> print(result["cvar_data"].R.shape)  # (3, 50)
    """

    return_mean = returns_dict["mean"]
    returns_data = returns_dict["returns"].to_numpy()
    num_scen = scenario_generation_settings["num_scen"]
    fit_type = scenario_generation_settings["fit_type"]

    if "kde_settings" in scenario_generation_settings:
        kde_settings = scenario_generation_settings["kde_settings"]
    else:
        kde_settings = {"device": "CPU", "bandwidth": 0.05, "kernel": "gaussian"}

    if fit_type == "gaussian":  # Gaussian distribution
        covariance = returns_dict["covariance"]
        R_log = np.random.multivariate_normal(return_mean, covariance, size=num_scen)
        R = np.transpose(R_log)
        p = np.ones(num_scen) / num_scen  # probability of each scenario

    elif fit_type == "kde":  # kde distribution
        R_log = generate_samples_kde(
            num_scen,
            returns_data,
            kde_settings=kde_settings,
            verbose=scenario_generation_settings["verbose"],
        )
        R = np.transpose(R_log)
        p = np.ones(num_scen) / num_scen  # probability of each scenario

    elif fit_type == "no_fit":  # use input data directly
        R = np.transpose(returns_data)
        num_scen = R.shape[1]
        p = np.ones(num_scen) / num_scen

    else:
        raise ValueError("Unsupported fit type: must be from gaussian, kde, or no_fit.")

    cvar_data = CvarData(mean=return_mean, R=R, p=p)

    returns_dict["cvar_data"] = cvar_data

    return returns_dict


def optimize_market_regimes(
    input_file_name: str,
    returns_compute_settings: dict,
    scenario_generation_settings: dict,
    all_regimes: dict,
    cvar_params: CvarParameters,
    solver_settings_list: list[dict],
    results_csv_file_name: str = None,
    num_synthetic: int = 0,
    print_results: bool = True,
):
    """
    Compare CVaR optimization performance across different regimes and solvers.

    Tests multiple solvers across different market regimes and collects
    performance metrics.

    Args:
        input_file_name (str): Path to input data file.
        returns_compute_settings (dict): Dictionary containing returns calculation settings.
        scenario_generation_settings (dict): Dictionary containing scenario generation settings.
        all_regimes (dict): Dictionary of regimes to test with format
            {'regime_name': regime_range}.
        cvar_params (CvarParameters): CVaR optimization parameters.
        solver_settings_list (list[dict]): List of solver settings to test.
            Each dict contains solver-specific settings
            (e.g., {'solver': cp.CLARABEL, 'verbose': False}).
        results_csv_file_name (str, optional): CSV filename to save results.
            Defaults to None.
        num_synthetic (int, optional): Number of synthetic data copies to generate.
            0 means no generation. Defaults to 0.
        print_results (bool, optional): Whether to print optimization results.
            Defaults to True.

    Returns:
        pd.DataFrame: Results dataframe with columns:
            - 'regime': Regime name
            - '{solver_name}-obj': Objective value for each solver
            - '{solver_name}-solve_time': Solve time for each solver
            - '{solver_name}-optimal_portfolio': Optimal portfolio for each solver
            - '{solver_name}-return': Expected return for each solver
            - '{solver_name}-CVaR': CVaR value for each solver

    Raises:
        FileNotFoundError: If problem_from_folder doesn't exist.
        ValueError: If solver_settings_list is empty.

    Example:
        >>> solver_settings_list = [
        ...     {'solver': cp.CLARABEL, 'verbose': False},
        ...     {'solver': cp.HIGHS, 'verbose': False},
        ... ]
        >>> results = optimize_market_regimes(
        ...     'sp500.csv', 'LOG', all_regimes, cvar_params, solver_settings_list
        ... )
    """
    from . import cvar_optimizer  # Lazy import

    if len(solver_settings_list) == 0:
        raise ValueError("Please provide at least one solver settings!")

    # Helper function to extract solver name from settings
    def get_solver_name(settings):
        """Extract solver name from solver settings dict."""
        if "solver" in settings:
            # CVXPY solver - extract name from solver object
            solver_obj = settings["solver"]
            return str(solver_obj).replace("cp.", "").replace("solvers.", "")
        else:
            raise ValueError(f"Unsupported solver settings: {settings}")

    # Build column names dynamically based on solvers
    columns = ["regime"]
    solver_names = []
    for settings in solver_settings_list:
        solver_name = get_solver_name(settings)
        solver_names.append(solver_name)
        columns.extend(
            [
                f"{solver_name}-obj",
                f"{solver_name}-solve_time",
                f"{solver_name}-return",
                f"{solver_name}-CVaR",
                f"{solver_name}-optimal_portfolio",
            ]
        )

    result_rows = []

    for regime_name, regime_range in all_regimes.items():
        print("=" * 70)
        print(f"Processing Regime: {regime_name}")
        print("=" * 70)

        # Create synthetic datasets on the fly if requested
        input_data_directory = (
            create_synthetic_stock_dataset(
                input_file_name, regime_name, regime_range, num_synthetic
            )
            if num_synthetic > 0
            else input_file_name
        )

        # create the returns_dict for the current regime
        curr_regime = {"name": regime_name, "range": regime_range}
        returns_dict = utils.calculate_returns(
            input_data_directory, curr_regime, returns_compute_settings
        )
        returns_dict = generate_cvar_data(returns_dict, scenario_generation_settings)

        # Initialize result row for this regime
        result_row = {"regime": regime_name}

        # Solve with each solver
        for idx, solver_settings in enumerate(solver_settings_list):
            solver_name = solver_names[idx]
            print(f"\n--- Testing Solver: {solver_name} ---")

            # Set up optimization problem
            cvar_problem = cvar_optimizer.CVaR(
                        returns_dict=returns_dict, cvar_params=cvar_params
            )
        
            # Solve optimization problem
            try:
                result, portfolio = cvar_problem.solve_optimization_problem(
                    solver_settings, print_results=print_results
                )

                # Store results with solver-specific column names
                result_row[f"{solver_name}-obj"] = result["obj"]
                result_row[f"{solver_name}-solve_time"] = result["solve time"]
                result_row[f"{solver_name}-return"] = result["return"]
                result_row[f"{solver_name}-CVaR"] = result["CVaR"]
                result_row[f"{solver_name}-optimal_portfolio"] = portfolio.print_clean(
                    verbose=False
                )

                print(
                    f"  ✓ {solver_name} - Objective: {result['obj']:.6f}, "
                    f"Time: {result['solve time']:.4f}s"
                    f"--------------------------------"
                )

            except Exception as e:
                print(f"  ✗ {solver_name} failed: {str(e)}")
                # Store None for failed solvers
                result_row[f"{solver_name}-obj"] = None
                result_row[f"{solver_name}-solve_time"] = None
                result_row[f"{solver_name}-return"] = None
                result_row[f"{solver_name}-CVaR"] = None
                result_row[f"{solver_name}-optimal_portfolio"] = None

        # Add this regime's results to list
        result_rows.append(result_row)

    # Create DataFrame from collected rows
    result_dataframe = pd.DataFrame(result_rows, columns=columns)

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print("\n")

    if results_csv_file_name:
        result_dataframe.to_csv(results_csv_file_name, index=False)
        print(f"Results saved to: {results_csv_file_name}")

    return result_dataframe


def create_synthetic_stock_dataset(
    training_directory: str, regime_name: str, regime_range: tuple, num_synthetic: int
):
    """Create synthetic stock dataset based on training data.

    Args:
        training_directory (str): Path to the training data directory.
        regime_name (str): Name of the market regime.
        regime_range (tuple): Date range for the regime (start_date, end_date).
        num_synthetic (int): Number of synthetic datasets to generate.

    Returns:
        str: Path to the saved synthetic dataset file.

    Raises:
        ValueError: If num_synthetic is less than or equal to 0.

    Example:
        >>> training_dir = "data/stock_data/sp500.csv"
        >>> regime = "bull_market"
        >>> date_range = ("2020-01-01", "2021-12-31")
        >>> save_path = create_synthetic_stock_dataset(
        ...     training_dir,
        ...     regime,
        ...     date_range,
        ...     num_synthetic=100
        ... )
        >>> print(save_path)  # data/stock_data/synthetic-bull_market-size_500.csv
    """
    if num_synthetic <= 0:
        raise ValueError("Please provide a valid integer for num_synthetic!")

    synthetic_data = scenario_generation.generate_synthetic_stock_data(
        dataset_directory=training_directory,
        num_synthetic=num_synthetic,
        fit_range=regime_range,
        generate_range=regime_range,
    )
    dataset_size = len(synthetic_data.columns)

    save_name = "synthetic-" + regime_name + f"-size_{dataset_size}.csv"
    save_path = os.path.join(os.path.dirname(training_directory), save_name)
    synthetic_data.to_csv(save_path)

    return save_path


def evaluate_portfolio_performance(
    cvar_data: CvarData,
    portfolio: Portfolio,
    confidence_level: float,
    covariance: np.ndarray,
):
    """Evaluate performance metrics for a given portfolio.

    Calculates expected return, variance, and CVaR for a non-optimized portfolio
    based on the provided CVaR data and confidence level.

    Args:
        cvar_data (CvarData): CVaR data containing mean returns and scenarios.
        portfolio (Portfolio): Portfolio object with weights and other attributes.
        confidence_level (float): Confidence level for CVaR calculation
            (e.g., 0.95).
        covariance (np.ndarray): Covariance matrix of asset returns.

    Returns:
        dict: Dictionary containing portfolio performance metrics with keys:
            - 'portfolio': Portfolio object
            - 'return': Expected portfolio return
            - 'variance': Portfolio variance
            - 'CVaR': Conditional Value at Risk

    Example:
        >>> import numpy as np
        >>> cvar_data = CvarData(
        ...     mean=np.array([0.08, 0.10, 0.12]),
        ...     R=np.random.randn(3, 100),
        ...     p=np.ones(100) / 100
        ... )
        >>> portfolio = Portfolio(tickers=["AAPL", "GOOGL", "MSFT"])
        >>> portfolio.weights = np.array([0.3, 0.4, 0.3])
        >>> covariance = np.eye(3) * 0.01
        >>> performance = evaluate_portfolio_performance(
        ...     cvar_data, portfolio, 0.95, covariance
        ... )
        >>> print(f"Return: {performance['return']:.4f}")
        >>> print(f"CVaR: {performance['CVaR']:.4f}")
    """
    portfolio_expected_return = portfolio.calculate_portfolio_expected_return(
        cvar_data.mean
    )
    portfolio_variance = portfolio.calculate_portfolio_variance(covariance)
    portfolio_CVaR = compute_CVaR(cvar_data, portfolio.weights, confidence_level)

    return {
        "portfolio": portfolio,
        "return": portfolio_expected_return,
        "variance": portfolio_variance,
        "CVaR": portfolio_CVaR,
    }


def compute_CVaR(cvar_data: CvarData, weights: np.ndarray, confidence_level: float):
    """Compute the Conditional Value at Risk (CVaR) of a portfolio.

    Calculates the expected value of losses beyond the Value at Risk (VaR)
    threshold for a given confidence level.

    Args:
        cvar_data (CvarData): CVaR data containing scenarios and probabilities.
        weights (np.ndarray): Portfolio weights vector.
        confidence_level (float): Confidence level for CVaR calculation (e.g., 0.95).

    Returns:
        float: CVaR value representing the expected loss beyond VaR.

    Example:
        >>> import numpy as np
        >>> cvar_data = CvarData(
        ...     mean=np.array([0.08, 0.10, 0.12]),
        ...     R=np.random.randn(3, 1000),
        ...     p=np.ones(1000) / 1000
        ... )
        >>> weights = np.array([0.4, 0.3, 0.3])
        >>> cvar_95 = compute_CVaR(cvar_data, weights, 0.95)
        >>> print(f"95% CVaR: {cvar_95:.4f}")
        95% CVaR: 0.0234
    """
    portfolio_returns = cvar_data.R.T @ weights
    VaR = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    tail_loss = portfolio_returns[portfolio_returns <= VaR]
    CVaR = np.abs(np.mean(tail_loss))

    return CVaR


def evaluate_single_asset_portfolios(cvar_problem):
    """Create DataFrame with performance metrics for single-asset portfolios.

    Evaluates the performance of portfolios where each portfolio consists of
    only one asset (stock) at maximum allowed weight.

    Args:
        cvar_problem: CVaR optimization problem object containing data,
            parameters, and ticker information.

    Returns:
        pd.DataFrame: DataFrame with index as ticker symbols and columns:
            ['portfolio', 'return', 'variance', 'CVaR'] containing performance
            metrics for each single-asset portfolio.

    Example:
        >>> from . import cvar_optimizer
        >>> # Assuming cvar_problem is already set up
        >>> single_asset_df = evaluate_single_asset_portfolios(cvar_problem)
        >>> print(single_asset_df.head())
                    portfolio     return  variance     CVaR
        AAPL      AAPL_single   0.1200   0.0400    0.0560
        GOOGL    GOOGL_single   0.1500   0.0500    0.0680
        MSFT      MSFT_single   0.1100   0.0350    0.0520
    """

    single_asset_portfolio_performance = pd.DataFrame(
        index=cvar_problem.tickers,
        columns=["portfolio", "return", "variance", "CVaR"],
    )

    for ticker_idx, ticker in enumerate(cvar_problem.tickers):
        portfolio_name = ticker + "_single_portfolio"
        weights_dict = {ticker: cvar_problem.params.w_max[ticker_idx]}
        cash = 1 - cvar_problem.params.w_max[ticker_idx]

        portfolio = Portfolio(
            tickers=cvar_problem.tickers, time_range=cvar_problem.regime_range
        )
        portfolio.portfolio_from_dict(portfolio_name, weights_dict, cash)

        portfolio_performance = evaluate_portfolio_performance(
            cvar_problem.data,
            portfolio,
            cvar_problem.params.confidence,
            cvar_problem.covariance,
        )
        # Assign each column explicitly to avoid dtype inference issues
        single_asset_portfolio_performance.loc[ticker, "portfolio"] = (
            portfolio_performance["portfolio"]
        )
        single_asset_portfolio_performance.loc[ticker, "return"] = (
            portfolio_performance["return"]
        )
        single_asset_portfolio_performance.loc[ticker, "variance"] = (
            portfolio_performance["variance"]
        )
        single_asset_portfolio_performance.loc[ticker, "CVaR"] = portfolio_performance[
            "CVaR"
        ]

    return single_asset_portfolio_performance


def generate_user_input_portfolios(
    portfolios_dict: dict, returns_dict: dict, existing_portfolios: list = None
):
    """Create Portfolio objects from user input dictionaries.

    Converts user-provided portfolio specifications into Portfolio objects
    and adds them to existing portfolios list.

    Args:
        portfolios_dict (dict): Dictionary of portfolio specifications with format:
            {portfolio_name: (weight_dict, cash_amount)}
        returns_dict (dict): Dictionary containing returns data and ticker
            information.
        existing_portfolios (list or pd.DataFrame, optional): Existing portfolios
            to append to. Can be a list of Portfolio objects or DataFrame with
            'portfolio' column. Defaults to empty list.

    Returns:
        list: List of Portfolio objects including existing and newly created
            portfolios.

    Raises:
        ValueError: If existing_portfolios type is not supported
            (must be list or DataFrame).

    Example:
        >>> portfolios_dict = {
        ...     "Tech_Heavy": ({"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.2}, 0.1),
        ...     "Equal_Weight": ({"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}, 0.0)
        ... }
        >>> returns_dict = {
        ...     "tickers": ["AAPL", "GOOGL", "MSFT"],
        ...     "regime": {"range": ("2020-01-01", "2021-12-31")}
        ... }
        >>> portfolios = generate_user_input_portfolios(portfolios_dict, returns_dict)
    """
    if existing_portfolios is None:
        existing_portfolios = []

    if isinstance(existing_portfolios, pd.DataFrame):
        if not existing_portfolios.empty:
            existing_portfolios = existing_portfolios["portfolio"].tolist()
        else:
            existing_portfolios = []

    elif isinstance(existing_portfolios, list):
        pass
    else:
        raise ValueError(
            "Existing portfolios type not supported - it has to be a list of "
            "Portfolios or a DataFrame with portfolio performance."
        )

    for portfolio_name, portfolio_tuple in portfolios_dict.items():
        weights_dict, cash = portfolio_tuple
        portfolio = Portfolio(
            tickers=returns_dict["tickers"], time_range=returns_dict["regime"]["range"]
        )
        portfolio.portfolio_from_dict(portfolio_name, weights_dict, cash)

        existing_portfolios.append(portfolio)

    return existing_portfolios


def evaluate_user_input_portfolios(
    cvar_problem,
    portfolios_dict: dict,
    returns_dict: dict,
    custom_portfolios=None,
):
    """Create DataFrame of portfolios with performance metrics.

    Evaluates user-provided portfolios and creates a DataFrame containing
    performance metrics for efficient frontier plotting or comparison.

    Args:
        cvar_problem: CVaR optimization problem object containing data
            and parameters.
        portfolios_dict (dict): Dictionary of portfolio specifications with format:
            {portfolio_name: (weight_dict, cash_amount)}
        returns_dict (dict): Dictionary containing returns data and ticker
            information.
        custom_portfolios (pd.DataFrame, optional): Existing custom portfolios
            DataFrame. Must have columns: ['portfolio_name', 'portfolio',
            'return', 'variance', 'CVaR']. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing portfolio performance metrics
            for all portfolios.

    Example:
        >>> portfolios_dict = {
        ...     "Conservative": ({"AAPL": 0.2, "GOOGL": 0.2}, 0.6),
        ...     "Aggressive": ({"AAPL": 0.5, "GOOGL": 0.4, "MSFT": 0.1}, 0.0)
        ... }
        >>> performance_df = evaluate_user_input_portfolios(
        ...     cvar_problem, portfolios_dict, returns_dict
        ... )
        >>> print(performance_df[['portfolio_name', 'return', 'CVaR']])
           portfolio_name    return     CVaR
        0  Conservative      0.0850   0.0320
        1  Aggressive        0.1150   0.0580
    """
    if custom_portfolios is None:
        custom_portfolios = pd.DataFrame(
            [], columns=["portfolio_name", "portfolio", "return", "variance", "CVaR"]
        )

    existing_portfolios = generate_user_input_portfolios(
        portfolios_dict, returns_dict, custom_portfolios
    )

    for portfolio in existing_portfolios:
        portfolio_performance = evaluate_portfolio_performance(
            cvar_problem.data,
            portfolio,
            cvar_problem.params.confidence,
            cvar_problem.covariance,
        )
        portfolio_performance["portfolio_name"] = portfolio.name

        portfolio_dataframe = (
            pd.Series(portfolio_performance, index=custom_portfolios.columns)
            .to_frame()
            .T
        )
        if custom_portfolios.shape[0] > 0:
            if portfolio.name not in custom_portfolios["portfolio_name"].values:
                custom_portfolios = pd.concat(
                    [custom_portfolios, portfolio_dataframe], ignore_index=False
                )
            else:
                print(
                    f"{portfolio_dataframe['portfolio_name'].values} already "
                    "exists or please change to a different portfolio name."
                )
        else:
            custom_portfolios = portfolio_dataframe

    custom_portfolios.reset_index(drop=True, inplace=True)

    return custom_portfolios


def create_efficient_frontier(
    returns_dict: dict,
    cvar_params: CvarParameters,
    solver_settings: dict,
    notional: float = 1e7,
    figsize: tuple = (12, 8),
    style: str = "publication",
    color_scheme: str = "modern",
    ra_num: int = 25,
    min_risk_aversion: float = -3,
    max_risk_aversion: float = 1,
    custom_portfolios_dict: dict = None,
    benchmark_portfolios: bool = True,
    show_discretized_portfolios: bool = True,
    discretization_params: dict = None,
    save_path: str = None,
    title: str = None,
    print_portfolio_results: bool = False,
    show_plot: bool = True,
    dpi: int = 300,
) -> tuple:
    """Create an efficient frontier plot with visualization features.

    This function generates an efficient frontier plot with styling,
    annotations, and portfolio analysis.

    Args:
        returns_dict (dict): Dictionary containing returns data and ticker
            information.
        cvar_params (CvarParameters): CVaR optimization parameters.
        solver_settings (dict): Solver configuration for optimization.
        notional (float, optional): Notional amount (in USD) for scaling
            returns display.
            Defaults to 1e7 (10 million USD).
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 8).
        style (str, optional): Plot style ("publication", "presentation", "minimal").
            Defaults to "publication".
        color_scheme (str, optional): Color scheme ("modern", "classic", "vibrant").
            Defaults to "modern".
        ra_num (int, optional): Number of risk aversion levels. Defaults to 25.
        min_risk_aversion (float, optional): Minimum risk aversion (log scale).
            Defaults to -3.
        max_risk_aversion (float, optional): Maximum risk aversion (log scale).
            Defaults to 1.
        custom_portfolios_dict (dict, optional): Custom portfolios to highlight.
            Format: {name: (weights_dict, cash)}. Defaults to None.
        benchmark_portfolios (bool, optional): Include benchmark portfolios
            (min variance, max Sharpe, max return). Defaults to True.
        show_discretized_portfolios (bool, optional): Show discretized
            portfolio combinations. Defaults to True.
        discretization_params (dict, optional): Parameters for discretized
            portfolios. Dict with keys: weight_discretization, max_assets,
            min_weight, max_weight, sum_to_one. Defaults to
            {"weight_discretization": 10, "max_assets": 5}.
        save_path (str, optional): Path to save the figure. Defaults to None.
        title (str, optional): Custom plot title. Defaults to auto-generated.
        print_portfolio_results (bool, optional): Whether to print the portfolio
            results. Defaults to False.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        dpi (int, optional): Resolution for saved figure. Defaults to 300.

    Returns:
        tuple: (results_df, fig, ax) containing the optimization results DataFrame,
            matplotlib figure, and axes objects.

    Example:
        >>> regime = {"name": "full_period", "range": ("2020-01-01", "2023-12-31")}
        >>> results_df, fig, ax = create_efficient_frontier(
        ...     returns_dict,
        ...     cvar_params,
        ...     {"solver": "CLARABEL", "verbose": False}
        ... )
    """
    from . import cvar_optimizer  # Lazy import

    if custom_portfolios_dict is None:
        custom_portfolios_dict = {}

    if discretization_params is None:
        discretization_params = {
            "weight_discretization": 10,
            "max_assets": 5,
            "min_weight": 0.0,
            "max_weight": 1.0,
            "sum_to_one": True,
        }

    # Color schemes
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
    colors = color_schemes[color_scheme]

    # Set style
    if style == "publication":
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=1.2)
    elif style == "presentation":
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("talk", font_scale=1.1)
    else:  # minimal
        plt.style.use("seaborn-v0_8-white")
        sns.set_context("notebook")

    # Initialize optimization problem
    cvar_problem = cvar_optimizer.CVaR(
        returns_dict=returns_dict, cvar_params=cvar_params
    )

    # Generate risk aversion range
    risk_aversion_list = np.logspace(
        start=min_risk_aversion, stop=max_risk_aversion, num=ra_num
    )[::-1]

    # Containers for results
    results_data = []
    portfolios = []

    print(f"Computing efficient frontier with {ra_num} portfolios...")

    for i, ra_value in enumerate(risk_aversion_list):
        cvar_problem.params.update_risk_aversion(ra_value)
        cvar_problem.risk_aversion_param.value = ra_value

        result_row, portfolio = cvar_problem.solve_optimization_problem(
            solver_settings, print_results=print_portfolio_results
        )

        result_row["risk_aversion"] = ra_value
        result_row["variance"] = portfolio.calculate_portfolio_variance(
            cvar_problem.covariance
        )
        result_row["volatility"] = np.sqrt(result_row["variance"])
        result_row["sharpe"] = (
            result_row["return"] / result_row["volatility"] * np.sqrt(252)
        )

        results_data.append(result_row)
        portfolios.append(portfolio)

        if (i + 1) % 10 == 0:
            print(f"   ✓ Completed {i + 1}/{ra_num} portfolios")

    # Create results DataFrame
    results_df = pd.DataFrame(results_data)

    # Identify key portfolios
    min_var_idx = results_df["variance"].idxmin()
    max_sharpe_idx = results_df["sharpe"].idxmax()
    max_return_idx = results_df["return"].idxmax()

    key_portfolios = {
        "Min Variance": min_var_idx,
        "Max Sharpe": max_sharpe_idx,
        "Max Return": max_return_idx,
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=colors["background"])
    ax.set_facecolor(colors["background"])

    # Plot efficient frontier with notional scaling and percentage CVaR
    ax.plot(
        results_df["CVaR"] * 100,  # Convert CVaR to percentage
        results_df["return"] * notional,  # Scale returns by notional
        linewidth=3,
        color=colors["frontier"],
        label="Efficient Frontier",
        zorder=3,
        alpha=0.9,
    )

    # Add gradient fill under the frontier
    ax.fill_between(
        results_df["CVaR"] * 100,  # Convert CVaR to percentage
        results_df["return"] * notional,  # Scale returns by notional
        alpha=0.1,
        color=colors["frontier"],
        zorder=1,
    )

    # Plot benchmark portfolios
    if benchmark_portfolios:
        benchmark_markers = ["o", "^", "s"]

        for i, (name, idx) in enumerate(key_portfolios.items()):
            ax.scatter(
                results_df.loc[idx, "CVaR"] * 100,  # Convert CVaR to percentage
                results_df.loc[idx, "return"] * notional,  # Scale returns by notional
                s=120,
                color=colors["benchmark"][i],
                marker=benchmark_markers[i],
                edgecolor="white",
                linewidth=2,
                label=name,
                zorder=4,
            )

            # Add annotations for key portfolios
            ax.annotate(
                f"{name}\nReturn: ${results_df.loc[idx, 'return'] * notional:,.0f}\n"
                + f"CVaR: {results_df.loc[idx, 'CVaR'] * 100:.1f}%",
                (
                    results_df.loc[idx, "CVaR"] * 100,
                    results_df.loc[idx, "return"] * notional,
                ),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=colors["benchmark"][i],
                    alpha=0.8,
                    edgecolor="white",
                ),
                fontsize=9,
                color="white",
                ha="left",
                zorder=5,
            )

    # Discretized portfolios (if requested)
    if show_discretized_portfolios:
        discretized_portfolios = evaluate_all_linear_combinations(
            returns_dict, cvar_params, **discretization_params
        )

        # Plot discretized portfolios with variance as hue
        scatter = ax.scatter(
            discretized_portfolios["CVaR"] * 100,  # Convert CVaR to percentage
            discretized_portfolios["return"] * notional,  # Scale returns by notional
            s=40,
            c=discretized_portfolios["variance"],
            cmap="plasma",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            label="Discretized Portfolios",
            zorder=2,
        )

        # Add colorbar for portfolio variance
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Portfolio Variance", rotation=270, labelpad=15)

    # Custom portfolios
    if custom_portfolios_dict:
        custom_portfolios = evaluate_user_input_portfolios(
            cvar_problem, custom_portfolios_dict, returns_dict
        )

        for _idx, row in custom_portfolios.iterrows():
            ax.scatter(
                row["CVaR"] * 100,  # Convert CVaR to percentage
                row["return"] * notional,  # Scale returns by notional
                s=100,
                color=colors["custom"],
                marker="D",
                edgecolor="white",
                linewidth=2,
                label=f"Custom: {row['portfolio_name']}",
                zorder=4,
            )

    # Styling and labels
    ax.set_xlabel(
        f"{cvar_params.confidence:.0%} CVaR (percentage)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Expected Return (${notional / 1e6:.0f}M Notional)",
        fontsize=12,
        fontweight="bold",
    )

    if title is None:
        title = f"Efficient Frontier - {ra_num} portfolios"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Grid and styling
    ax.grid(True, alpha=0.3, color=colors["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E0E0E0")
    ax.spines["bottom"].set_color("#E0E0E0")

    # Legend
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        fontsize=10,
    )

    plt.tight_layout()

    # Save the figure
    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=colors["background"],
            edgecolor="none",
        )
        print(f"💾 Plot saved to: {save_path}")

    # Show the plot
    if show_plot:
        plt.show()

    print("Efficient frontier analysis complete!")

    return results_df, fig, ax


def evaluate_all_linear_combinations(
    returns_dict: dict,
    cvar_params: CvarParameters,
    weight_discretization: int = 20,
    max_assets: int = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    use_gpu: bool = True,
):
    """
    Discretize dataset and evaluate all linear combinations of stocks for
    returns and CVaR using GPU acceleration and parallel computing.

    This function creates discrete weight combinations for all stocks and
    evaluates portfolio performance using vectorized operations with CuPy Numeric
    for enhanced NumPy compatibility and potential GPU acceleration.

    **Mean Return Constraints:**
    - Assets with negative mean returns: only zero weight allowed (no long positions)
    - Assets with positive mean returns: full weight range allowed (long positions only)

    Args:
        returns_dict (dict): Dictionary containing returns data and market information.
        cvar_params (CvarParameters): CVaR optimization parameters.
        weight_discretization (int, optional): Number of discrete weight
            levels for each asset. Defaults to 20.
        max_assets (int, optional): Maximum number of assets to consider.
            If None, uses all assets. Defaults to None.
        min_weight (float, optional): Minimum weight for any asset.
            Defaults to 0.0.
        max_weight (float, optional): Maximum weight for any asset.
            Defaults to 1.0.
        use_gpu (bool, optional): Whether to use GPU acceleration with CuPy.
            When available, CuPy Numeric is used for all operations for
            better performance. Defaults to True.

        Returns:
        pd.DataFrame: DataFrame containing all portfolio combinations with
            their performance metrics:
            - 'weights': Portfolio weights as a dictionary
            - 'return': Expected portfolio return
            - 'variance': Portfolio variance
            - 'volatility': Portfolio volatility (standard deviation)
            - 'CVaR': Conditional Value at Risk
            - 'sharpe': Sharpe ratio (return/CVaR)
            - 'num_assets': Number of non-zero assets in portfolio

    Raises:
        ValueError: If parameters are invalid or incompatible.

    Example:
        >>> results = evaluate_all_linear_combinations(
        ...     returns_dict,
        ...     cvar_params,
        ...     weight_discretization=10,
        ...     max_assets=5,
        ...     use_gpu=True
        ... )
    """
    if weight_discretization < 2:
        raise ValueError("weight_discretization must be at least 2")

    # Try to import CuPy Numeric for enhanced performance
    try:
        import cupynumeric as cnp

        cupynumeric_available = True
        print("Using CuPy Numeric for enhanced NumPy operations")
    except ImportError:
        import numpy as cnp  # Fallback to regular NumPy

        cupynumeric_available = False
        print("CuPy Numeric not available, using standard NumPy")

    # Determine GPU acceleration separately
    gpu_available = use_gpu and cupynumeric_available

    # Extract data
    cvar_data = returns_dict["cvar_data"]
    covariance = returns_dict["covariance"]
    tickers = returns_dict["tickers"]

    if max_assets is None:
        max_assets = len(tickers)
    max_assets = min(max_assets, len(tickers))

    # Validate constraint feasibility before generating combinations
    # Only count assets with positive mean returns for weight sum calculations
    mean_returns = cvar_data.mean[:max_assets]
    positive_mean_assets = sum(1 for i in range(max_assets) if mean_returns[i] >= 0)
    negative_mean_assets = max_assets - positive_mean_assets

    # Calculate actual possible weight sums based on mean return constraints
    min_possible_weight_sum = (
        positive_mean_assets * min_weight
    )  # negative mean assets contribute 0
    max_possible_weight_sum = (
        positive_mean_assets * max_weight
    )  # negative mean assets contribute 0
    min_required_weight_sum = 1.0 - cvar_params.c_max
    max_allowed_weight_sum = 1.0 - cvar_params.c_min

    print("Mean return analysis:")
    print(f"  Assets with positive mean returns: {positive_mean_assets}")
    print(
        f"  Assets with negative mean returns: {negative_mean_assets} "
        "(will have zero weight)"
    )
    print("Constraint validation:")
    print(
        f"  Possible weight sum range: "
        f"[{min_possible_weight_sum:.3f}, {max_possible_weight_sum:.3f}]"
    )
    print(
        f"  Required weight sum range: "
        f"[{min_required_weight_sum:.3f}, {max_allowed_weight_sum:.3f}]"
    )

    # Check if constraints are feasible
    if min_possible_weight_sum > max_allowed_weight_sum:
        raise ValueError(
            f"Impossible constraints: minimum possible weight sum "
            f"({min_possible_weight_sum:.3f}) exceeds maximum allowed "
            f"({max_allowed_weight_sum:.3f}). "
            f"Try reducing w_min ({min_weight}) or "
            f"increasing c_min ({cvar_params.c_min}). "
            f"Note: {negative_mean_assets} assets with negative mean returns "
            "are excluded from long positions."
        )

    if max_possible_weight_sum < min_required_weight_sum:
        raise ValueError(
            f"Impossible constraints: maximum possible weight sum "
            f"({max_possible_weight_sum:.3f}) is below minimum required "
            f"({min_required_weight_sum:.3f}). "
            f"Try increasing w_max ({max_weight}) or "
            f"reducing c_max ({cvar_params.c_max}). "
            f"Note: Only {positive_mean_assets} assets with positive "
            "mean returns can have non-zero weights."
        )

    # Create discrete weight levels based on mean returns
    # Assets with negative mean returns: only allow 0 weight
    # Assets with positive mean returns: allow min_weight to max_weight

    mean_returns = cvar_data.mean[:max_assets]
    asset_weight_levels = []

    for i in range(max_assets):
        if mean_returns[i] < 0:
            # Negative mean return: only allow zero weight
            if gpu_available:
                levels = cnp.array([0.0])
            else:
                levels = np.array([0.0])
            asset_name = tickers[i] if i < len(tickers) else f"Asset_{i}"
            print(
                f"Asset {i} ({asset_name}): negative mean return, "
                "only zero weight allowed"
            )
        else:
            # Positive mean return: allow full weight range
            if gpu_available:
                levels = cnp.linspace(min_weight, max_weight, weight_discretization)
            else:
                levels = np.linspace(min_weight, max_weight, weight_discretization)
            asset_name = tickers[i] if i < len(tickers) else f"Asset_{i}"
            print(
                f"Asset {i} ({asset_name}): positive mean return, "
                "full weight range allowed"
            )

        asset_weight_levels.append(levels)

    if gpu_available:
        print("Using CuPy Numeric for weight generation")
    else:
        print("Using standard NumPy for weight generation")

    # Calculate total combinations (product of lengths of each asset's weight levels)
    total_combinations = 1
    for levels in asset_weight_levels:
        total_combinations *= len(levels)

    print(
        f"Generating {total_combinations:,} combinations "
        "based on mean return constraints..."
    )

    # Use meshgrid for efficient combination generation
    # with asset-specific weight levels
    if gpu_available:
        grids = cnp.meshgrid(*asset_weight_levels, indexing="ij")
        all_weights = cnp.stack([grid.ravel() for grid in grids], axis=1)
        weight_sums = cnp.sum(all_weights, axis=1)
    else:
        grids = np.meshgrid(*asset_weight_levels, indexing="ij")
        all_weights = np.stack([grid.ravel() for grid in grids], axis=1)
        weight_sums = np.sum(all_weights, axis=1)

    # Allow for some flexibility in the constraints
    tolerance = 1e-6
    valid_mask = (weight_sums >= min_required_weight_sum - tolerance) & (
        weight_sums <= max_allowed_weight_sum + tolerance
    )

    valid_weights = all_weights[valid_mask]
    valid_combinations = len(valid_weights)

    if valid_combinations == 0:
        # Provide detailed error message
        if gpu_available:
            actual_min = cnp.min(weight_sums)
            actual_max = cnp.max(weight_sums)
        else:
            actual_min = np.min(weight_sums)
            actual_max = np.max(weight_sums)
        raise ValueError(
            f"No valid weight combinations found. "
            f"Generated weight sums range: "
            f"[{actual_min:.3f}, {actual_max:.3f}], "
            f"but required range is: "
            f"[{min_required_weight_sum:.3f}, {max_allowed_weight_sum:.3f}]. "
            f"Try adjusting weight bounds "
            f"(w_min={min_weight}, w_max={max_weight}) "
            f"or cash constraints "
            f"(c_min={cvar_params.c_min}, c_max={cvar_params.c_max})"
        )

    print(f"Found {valid_combinations:,} valid combinations after filtering")

    # Move data to GPU if available
    if gpu_available:
        try:
            print("Moving data to GPU...")
            valid_weights_gpu = cnp.asarray(valid_weights)
            # Create copies of sliced arrays to avoid view issues with cuPyNumeric
            mean_returns_gpu = cnp.asarray(cvar_data.mean[:max_assets].copy())
            covariance_gpu = cnp.asarray(covariance[:max_assets, :max_assets].copy())
            scenarios_gpu = cnp.asarray(cvar_data.R[:max_assets, :].copy())
        except Exception as e:
            print(f"GPU memory error: {e}. Falling back to CPU.")
            gpu_available = False

    if not gpu_available:
        # Use standard NumPy for CPU fallback
        valid_weights_gpu = valid_weights
        mean_returns_gpu = cvar_data.mean[:max_assets]
        covariance_gpu = covariance[:max_assets, :max_assets]
        scenarios_gpu = cvar_data.R[:max_assets, :]

    # Process all portfolios at once using vectorized operations
    print(f"Processing {valid_combinations:,} portfolios...")

    # Vectorized calculations for all portfolios
    if gpu_available:
        # GPU calculations
        portfolio_returns = cnp.dot(valid_weights_gpu, mean_returns_gpu)
        temp = cnp.dot(valid_weights_gpu, covariance_gpu)
        portfolio_variances = cnp.sum(temp * valid_weights_gpu, axis=1)
        portfolio_returns_scenarios = cnp.dot(valid_weights_gpu, scenarios_gpu)

        # Calculate CVaR for each portfolio
        portfolio_cvars = cnp.zeros(valid_combinations)
        confidence_percentile = (1 - cvar_params.confidence) * 100

        for i in range(valid_combinations):
            scenario_returns = portfolio_returns_scenarios[i]
            var_threshold = cnp.percentile(scenario_returns, confidence_percentile)
            tail_losses = scenario_returns[scenario_returns <= var_threshold]
            if len(tail_losses) > 0:
                portfolio_cvars[i] = cnp.abs(cnp.mean(tail_losses))
            else:
                portfolio_cvars[i] = 0.0

        # Move results back to CPU
        weights_cpu = cnp.asnumpy(valid_weights_gpu)
        returns_cpu = cnp.asnumpy(portfolio_returns)
        variances_cpu = cnp.asnumpy(portfolio_variances)
        cvars_cpu = cnp.asnumpy(portfolio_cvars)
    else:
        # CPU calculations using standard NumPy
        portfolio_returns = np.dot(valid_weights_gpu, mean_returns_gpu)
        temp = np.dot(valid_weights_gpu, covariance_gpu)
        portfolio_variances = np.sum(temp * valid_weights_gpu, axis=1)
        portfolio_returns_scenarios = np.dot(valid_weights_gpu, scenarios_gpu)

        # Calculate CVaR for each portfolio
        portfolio_cvars = np.zeros(valid_combinations)
        confidence_percentile = (1 - cvar_params.confidence) * 100

        for i in range(valid_combinations):
            scenario_returns = portfolio_returns_scenarios[i]
            var_threshold = np.percentile(scenario_returns, confidence_percentile)
            tail_losses = scenario_returns[scenario_returns <= var_threshold]
            if len(tail_losses) > 0:
                portfolio_cvars[i] = np.abs(np.mean(tail_losses))
            else:
                portfolio_cvars[i] = 0.0

        weights_cpu = valid_weights_gpu
        returns_cpu = portfolio_returns
        variances_cpu = portfolio_variances
        cvars_cpu = portfolio_cvars

    # Calculate derived metrics
    if gpu_available:
        volatilities = cnp.sqrt(variances_cpu)
        sharpe_ratios = cnp.where(cvars_cpu > 0, returns_cpu / cvars_cpu, 0.0)
        num_assets = cnp.sum(weights_cpu > 1e-10, axis=1)
    else:
        volatilities = np.sqrt(variances_cpu)
        sharpe_ratios = np.where(cvars_cpu > 0, returns_cpu / cvars_cpu, 0.0)
        num_assets = np.sum(weights_cpu > 1e-10, axis=1)

    # Normalize weights and calculate cash for each portfolio
    if gpu_available:
        weight_sums = cnp.sum(weights_cpu, axis=1)
        cash_amounts = cnp.maximum(0, 1.0 - weight_sums)
        total_sums = weight_sums + cash_amounts
    else:
        weight_sums = np.sum(weights_cpu, axis=1)
        cash_amounts = np.maximum(0, 1.0 - weight_sums)
        total_sums = weight_sums + cash_amounts

    # Create results list
    results_list = []
    for i in range(valid_combinations):
        weights_raw = weights_cpu[i]
        cash_raw = cash_amounts[i]
        total_sum = total_sums[i]

        # Normalize to sum to 1
        normalized_weights = weights_raw / total_sum
        normalized_cash = cash_raw / total_sum

        # Create weights dictionary (only include first max_assets)
        weights_dict = {
            tickers[j]: float(normalized_weights[j])
            for j in range(min(max_assets, len(tickers)))
        }

        result_row = {
            "combination_id": i,
            "weights": weights_dict,
            "weights_array": normalized_weights.copy(),
            "return": float(returns_cpu[i]),
            "variance": float(variances_cpu[i]),
            "volatility": float(volatilities[i]),
            "CVaR": float(cvars_cpu[i]),
            "sharpe": float(sharpe_ratios[i]),
            "num_assets": int(num_assets[i]),
            "cash": float(normalized_cash),
        }

        results_list.append(result_row)

    print(f"Completed processing {valid_combinations:,} portfolios")

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    # Add ranking columns
    results_df["return_rank"] = results_df["return"].rank(ascending=False)
    results_df["cvar_rank"] = results_df["CVaR"].rank(
        ascending=True
    )  # Lower CVaR is better
    results_df["sharpe_rank"] = results_df["sharpe"].rank(ascending=False)

    # Sort by Sharpe ratio (descending)
    results_df = results_df.sort_values("sharpe", ascending=False).reset_index(
        drop=True
    )

    return results_df


def normalize_portfolio_weights_to_one(weights_dict: dict, cash: float):
    """
    Normalize portfolio weights and cash to sum to 1.

    Args:
        weights_dict (dict): Dictionary mapping tickers to portfolio weights.
        cash (float): Portfolio cash amount.

    Returns:
        tuple: (normalized_weights_dict, normalized_cash) where:
            - normalized_weights_dict (dict): Normalized portfolio weights
            - normalized_cash (float): Normalized portfolio cash

    Example:
        >>> weights_dict = {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.2}
        >>> cash = 0.2
        >>> normalized_weights, normalized_cash = normalize_portfolio_weights_to_one(
        ...     weights_dict, cash
        ... )
        >>> print(normalized_weights)
        {'AAPL': 0.272..., 'GOOGL': 0.363..., 'MSFT': 0.181...}
        >>> print(normalized_cash)  # 0.181...
        >>> # Verify sum equals 1
        >>> total = sum(normalized_weights.values()) + normalized_cash
        >>> print(f"{total:.10f}")  # 1.0000000000
    """
    weights = np.array(list(weights_dict.values()))
    raw_sum = np.sum(weights) + cash
    normalized_weights = weights / raw_sum
    normalized_cash = cash / raw_sum
    normalized_weights_dict = {
        ticker: weight
        for ticker, weight in zip(weights_dict.keys(), normalized_weights)
    }
    normalized_cash = normalized_cash
    return normalized_weights_dict, normalized_cash


def compare_cvxpy_vs_cuopt(
    returns_dict: dict,
    cvar_params: CvarParameters,
    cvxpy_solver_settings: dict = None,
    cuopt_solver_settings: dict = None,
    print_results: bool = True,
):
    """
    Compare CVXPY and cuOpt implementations for setup time and solve results.

    Creates separate CVaR optimizer instances for each API to compare performance.

    Parameters
    ----------
    returns_dict : dict
        Input data containing regime info and CvarData instance
    cvar_params : CvarParameters
        Constraint parameters and optimization settings
    cvxpy_solver_settings : dict, optional
        Solver settings for CVXPY
    cuopt_solver_settings : dict, optional
        Solver settings for cuOpt
    print_results : bool, default True
        Whether to print comparison results

    Returns
    -------
    dict
        Comparison results including setup times, solve times, and solution differences

    Examples
    --------
    >>> import cvxpy as cp
    >>> # Prepare data and parameters
    >>> regime = {"name": "bull_market", "range": ("2020-01-01", "2021-12-31")}
    >>> returns_dict = calculate_returns(
    ...     "data/stock_data/sp500.csv", regime, "LOG", cvar_params
    ... )
    >>> cvar_params = CvarParameters(num_scen=100, confidence=0.95)
    >>>
    >>> # Compare CVXPY and cuOpt
    >>> cvxpy_settings = {"solver": cp.CLARABEL, "verbose": False}
    >>> cuopt_settings = {"api": "cuopt_python", "verbose": False}
    >>> results = compare_cvxpy_vs_cuopt(
    ...     returns_dict,
    ...     cvar_params,
    ...     cvxpy_settings,
    ...     cuopt_settings,
    ...     print_results=True
    ... )
    >>>
    >>> # Access comparison results
    >>> print(f"cuOpt speedup: {results['comparison']['total_speedup']:.2f}x")
    cuOpt speedup: 15.34x
    >>> print(f"Objective difference: {results['comparison']['objective_diff']:.8f}")
    Objective difference: 0.00000123
    >>> print(f"Max weight diff: {results['comparison']['max_weight_diff']:.8f}")
    Max weight diff: 0.00000045
    """
    from . import cvar_optimizer  # Lazy import

    if cvxpy_solver_settings is None:
        cvxpy_solver_settings = {}
    if cuopt_solver_settings is None:
        cuopt_solver_settings = {}

    print(f"{'=' * 70}")
    print("CVXPY vs cuOpt Performance Comparison")
    print(f"{'=' * 70}")

    results = {}
    cvxpy_portfolio = None
    cuopt_portfolio = None

    try:
        # ===============================
        # CVXPY Setup and Solve
        # ===============================
        print("\nCreating CVXPY optimizer instance...")
        cvxpy_optimizer = cvar_optimizer.CVaR(
            returns_dict=returns_dict, cvar_params=cvar_params, api_choice="cvxpy"
        )

        print("Solving with CVXPY...")
        cvxpy_result_row, cvxpy_portfolio = cvxpy_optimizer.solve_optimization_problem(
            cvxpy_solver_settings, print_results=False
        )
        cvxpy_setup_time = cvxpy_optimizer.set_up_time
        cvxpy_solve_time = cvxpy_result_row["solve time"]

        # Store CVXPY results
        cvxpy_objective = cvxpy_optimizer.optimization_problem.value
        cvxpy_status = cvxpy_optimizer.optimization_problem.status

        if cvxpy_portfolio is None:
            print(
                f"Warning: CVXPY optimization failed or returned no solution. "
                f"Status: {cvxpy_status}"
            )

        results["cvxpy"] = {
            "setup_time": cvxpy_setup_time,
            "solve_time": cvxpy_solve_time,
            "total_time": cvxpy_setup_time + cvxpy_solve_time,
            "portfolio": cvxpy_portfolio,
            "objective_value": cvxpy_objective,
            "status": cvxpy_status,
        }

        # ===============================
        # cuOpt Setup and Solve
        # ===============================
        print("\nCreating cuOpt optimizer instance...")
        cuopt_optimizer = cvar_optimizer.CVaR(
            returns_dict=returns_dict,
            cvar_params=cvar_params,
            api_choice="cuopt_python",
        )

        print("Solving with cuOpt...")
        cuopt_result_row, cuopt_portfolio = cuopt_optimizer.solve_optimization_problem(
            cuopt_solver_settings, print_results=False
        )
        cuopt_solve_time = cuopt_result_row["solve time"]
        cuopt_setup_time = cuopt_optimizer.set_up_time

        # Store cuOpt results
        cuopt_objective = cuopt_optimizer._cuopt_problem.ObjValue
        cuopt_status = cuopt_optimizer._cuopt_problem.Status.name

        if cuopt_portfolio is None:
            print(
                f"Warning: cuOpt optimization failed or returned no solution. "
                f"Status: {cuopt_status}"
            )

        results["cuopt"] = {
            "setup_time": cuopt_setup_time,
            "solve_time": cuopt_solve_time,
            "total_time": cuopt_setup_time + cuopt_solve_time,
            "portfolio": cuopt_portfolio,
            "objective_value": cuopt_objective,
            "status": cuopt_status,
        }

        # ===============================
        # Calculate Differences
        # ===============================
        setup_speedup = (
            cvxpy_setup_time / cuopt_setup_time
            if cuopt_setup_time > 0
            else float("inf")
        )
        solve_speedup = (
            cvxpy_solve_time / cuopt_solve_time
            if cuopt_solve_time > 0
            else float("inf")
        )
        total_speedup = (cvxpy_setup_time + cvxpy_solve_time) / (
            cuopt_setup_time + cuopt_solve_time
        )

        # Portfolio weight differences (only if both portfolios exist)
        if cvxpy_portfolio is not None and cuopt_portfolio is not None:
            weight_diff = np.abs(cvxpy_portfolio.weights - cuopt_portfolio.weights)
            max_weight_diff = np.max(weight_diff)
            mean_weight_diff = np.mean(weight_diff)
        else:
            max_weight_diff = float("inf")
            mean_weight_diff = float("inf")

        # Objective value difference
        if cvxpy_objective is not None and cuopt_objective is not None:
            obj_diff = abs(cvxpy_objective - cuopt_objective)
            obj_rel_diff = (
                obj_diff / abs(cvxpy_objective) * 100
                if cvxpy_objective != 0
                else float("inf")
            )
        else:
            obj_diff = float("inf")
            obj_rel_diff = float("inf")

        results["comparison"] = {
            "setup_speedup": setup_speedup,
            "solve_speedup": solve_speedup,
            "total_speedup": total_speedup,
            "max_weight_diff": max_weight_diff,
            "mean_weight_diff": mean_weight_diff,
            "objective_diff": obj_diff,
            "objective_rel_diff_pct": obj_rel_diff,
        }

        if print_results:
            _print_comparison_results(results)

    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        results["error"] = str(e)

    return results


def _print_comparison_results(results):
    """Print formatted comparison results.

    Args:
        results (dict): Results dictionary from compare_cvxpy_vs_cuopt() containing
            'cvxpy', 'cuopt', and 'comparison' keys with timing and solution data.
    """
    cvxpy = results["cvxpy"]
    cuopt = results["cuopt"]
    comp = results["comparison"]

    print(f"\n{'=' * 70}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'=' * 70}")

    # Timing comparison table
    print("\nTIMING COMPARISON")
    print(f"{'-' * 50}")
    print(f"{'Metric':<20} {'CVXPY':<12} {'cuOpt':<12} {'Speedup':<10}")
    print(f"{'-' * 50}")
    print(
        f"{'Setup Time':<20} {cvxpy['setup_time']:<12.4f} "
        f"{cuopt['setup_time']:<12.4f} {comp['setup_speedup']:<10.2f}x"
    )
    print(
        f"{'Solve Time':<20} {cvxpy['solve_time']:<12.4f} "
        f"{cuopt['solve_time']:<12.4f} {comp['solve_speedup']:<10.2f}x"
    )
    print(
        f"{'Total Time':<20} {cvxpy['total_time']:<12.4f} "
        f"{cuopt['total_time']:<12.4f} {comp['total_speedup']:<10.2f}x"
    )

    # Solution quality comparison
    print("\nSOLUTION QUALITY COMPARISON")
    print(f"{'-' * 50}")
    print(f"{'Status':<25} CVXPY: {cvxpy['status']:<15} cuOpt: {cuopt['status']}")

    # Handle objective values that might be None
    cvxpy_obj_str = (
        f"{cvxpy['objective_value']:.6f}"
        if cvxpy["objective_value"] is not None
        else "N/A"
    )
    cuopt_obj_str = (
        f"{cuopt['objective_value']:.6f}"
        if cuopt["objective_value"] is not None
        else "N/A"
    )
    print(f"{'Objective Value':<25} CVXPY: {cvxpy_obj_str:<15} cuOpt: {cuopt_obj_str}")

    # Handle differences that might be infinite
    if comp["objective_diff"] == float("inf"):
        print(f"{'Objective Difference':<25} N/A (one solver failed)")
    else:
        print(
            f"{'Objective Difference':<25} {comp['objective_diff']:.8f} "
            f"({comp['objective_rel_diff_pct']:.4f}%)"
        )

    if comp["max_weight_diff"] == float("inf"):
        print(f"{'Max Weight Difference':<25} N/A (one solver failed)")
        print(f"{'Mean Weight Difference':<25} N/A (one solver failed)")
    else:
        print(f"{'Max Weight Difference':<25} {comp['max_weight_diff']:.8f}")
        print(f"{'Mean Weight Difference':<25} {comp['mean_weight_diff']:.8f}")

    # Summary
    print("\nSUMMARY")
    print(f"{'-' * 50}")
    if comp["total_speedup"] > 1:
        print(f"cuOpt is {comp['total_speedup']:.2f}x faster overall")
    else:
        print(f"CVXPY is {1 / comp['total_speedup']:.2f}x faster overall")

    # Only compare solutions if both solvers succeeded
    if comp["objective_rel_diff_pct"] == float("inf"):
        print("Cannot compare solution quality - one or both solvers failed")
    elif comp["objective_rel_diff_pct"] < 0.01:
        print("Solutions match within 0.01% tolerance")
    elif comp["objective_rel_diff_pct"] < 1.0:
        print(f"Solutions differ by {comp['objective_rel_diff_pct']:.4f}%")
    else:
        print(f"Significant solution difference: {comp['objective_rel_diff_pct']:.4f}%")

    print(f"{'=' * 70}\n")
