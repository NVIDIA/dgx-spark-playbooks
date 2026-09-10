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

"""Utility functions for portfolio optimization and data processing."""

import os

from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


def get_input_data(filepath):
    """Load input data from file."""
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    if file_extension == ".csv":
        df = pd.read_csv(filepath, index_col=0)
    elif file_extension == ".parquet":
        df = pd.read_parquet(filepath)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath)
    elif file_extension == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    df = df.dropna(axis=1)
    return df


def calculate_returns(
    input_dataset: Union[pd.DataFrame, str],
    regime_dict: dict = None,
    returns_compute_settings: Union[dict, str] = None,
):
    """
    preprocess the dat from a particular period of time.
    Assuming the returns are log normally distributed, return the mean and
    covariance of the log returns and the log returns

    Parameters:
    :input_dataset: pandas DataFrame or the path to the input dataset
    :return_type: str, type of the returns. For example, "LOG" means log returns,
            "PNL" means the dataset is already in the format of P&L data.
            "NORMAL" means absolute returns.
    :regime_dict: dict of the format {'name': , 'range':(start, end)}
    :returns_compute_settings: Union[dict, str], dictionary containing returns calculation settings or the return type.
            If a string is provided, it is the return type.
            If a dictionary is provided, it contains the following keys:
            - "return_type": str, type of the returns. For example, "LOG" means log returns,
            - "freq": int, frequency of the returns. For example, freq = 1 means daily returns.
            - "returns_compute_device": str, device to use for returns calculation. For example, "GPU" or "CPU".
            - "verbose": bool, whether to print verbose output.
    """
    # set the default values for the returns calculation settings
    if returns_compute_settings.get("returns_compute_device") is None:
        returns_compute_settings["returns_compute_device"] = "CPU"
    if returns_compute_settings.get("verbose") is None:
        returns_compute_settings["verbose"] = False
    if returns_compute_settings.get("freq") is None:
        returns_compute_settings["freq"] = 1
    if returns_compute_settings.get("return_type") is None:
        returns_compute_settings["return_type"] = "LOG"

    return_type = returns_compute_settings["return_type"].upper()
    freq = returns_compute_settings["freq"]

    if isinstance(input_dataset, str):
        input_data = get_input_data(input_dataset)
    else:
        input_data = input_dataset

    if regime_dict is None:
        input_data = input_data
    else:
        start, end = regime_dict["range"]
        input_data = input_data.loc[start:end]
        
    input_data = input_data.dropna(axis=1)

    if return_type == "LOG":
        returns_dataframe = calculate_log_returns(input_data, freq)
    elif return_type == "PNL":
        returns_dataframe = input_data
    elif return_type == "NORMAL":
        returns_dataframe = compute_abs_returns(input_data, freq)
    else:
        raise NotImplementedError("Invalid return type!")

    returns_array = returns_dataframe.to_numpy()
    m = np.mean(returns_array, axis=0)
    cov = np.cov(returns_array.transpose())

    returns_dict = {
        "return_type": return_type,
        "returns": returns_dataframe,
        "regime": regime_dict,
        "dates": returns_dataframe.index,
        "mean": m,
        "covariance": cov,
        "tickers": list(input_data.columns),
    }

    return returns_dict


def calculate_log_returns(price_data, freq=1):
    """compute the log returns given a price dataframe"""
    # compute the log returns
    returns_dataframe = price_data.apply(np.log) - price_data.shift(freq).apply(np.log)
    returns_dataframe = returns_dataframe.dropna(how="all")
    returns_dataframe = returns_dataframe.fillna(0)

    return returns_dataframe


def compute_abs_returns(price_data, freq=1):
    """
    compute the absolute returns using freq. For example, freq = 1 means today - yesterday.
    """
    returns_dataframe = price_data.diff(freq)
    returns_dataframe = returns_dataframe.dropna(how="all")
    returns_dataframe = returns_dataframe.fillna(0)

    return returns_dataframe


def plot_efficient_frontier(
    risk_measure,
    result_dataframe,
    single_asset_portfolio,
    custom_portfolios,
    key_portfolios,
    verbose=False,
    title=None,
    show_plot=True,
    EF_plot_png_name=None,
    notional=1e7,
):
    """
    plot the efficient frontier using the optimization results of different
    risk-aversion levels in Seaborn.

    Parameters:
    :risk_measure: str
    :result_dataframe: Pandas DataFrame - (num_risks_levels, ?) where each row
        records the result of the optimization w.r.t. a certain risk level
    :single_asset_portfolio: Pandas DataFrame - (n_assets, #performance metrics)
        each row records the performance of the portfolio made up of one single asset
    :key_portfolios: dict - {portfolio_name: marker} of names of the portfolios
        (and corresponding markers) to highlight on the efficient frontier
        (e.g. min var, max Sharpe, max return, etc.)
    :custom_portfolios: Pandas DataFrame - (#user inputs, #performance metrics)
        each row records the performance of a custom portfolio from user input
    :show_plot: bool - whether to show plot
    :EF_plot_png_name: str - save the figure under the name EF_plot_png_name
    """
    # Apply consistent styling
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=0.9)
    sns.set_palette(palette="Blues_d")
    plt.figure(figsize=(10, 7), dpi=300)

    # Create scaled versions of the data for plotting
    result_dataframe_scaled = result_dataframe.copy()
    result_dataframe_scaled[f"{risk_measure}_percent"] = (
        result_dataframe_scaled[risk_measure] * 100
    )
    result_dataframe_scaled["return_scaled"] = (
        result_dataframe_scaled["return"] * notional
    )

    if key_portfolios is not None:
        # plot the markers for the key portfolios
        example_portfolio = pd.DataFrame({}, columns=result_dataframe.columns)
        for portfolio_name, marker in key_portfolios.items():
            portfolio_idx = get_portfolio(result_dataframe, portfolio_name)
            example_portfolio = pd.concat(
                [example_portfolio, result_dataframe.iloc[portfolio_idx].to_frame().T]
            )
            portfolio_data_scaled = (
                result_dataframe_scaled.iloc[portfolio_idx].to_frame().T
            )
            sns.scatterplot(
                data=portfolio_data_scaled,
                x=f"{risk_measure}_percent",
                y="return_scaled",
                marker=marker,
                s=100,
                color="darkorange",
                label=portfolio_name,
                legend=True,
                zorder=2,
            )
        example_portfolio = example_portfolio.reset_index()

        if verbose:
            # create the annotation box for the key portfolios
            _ = []  # annotated_points (unused)
            _ = []  # annotation_list (unused)

            offset_list = [(-15, -150), (20, -70), (-15, -70)]

            for row_idx, row in example_portfolio.iterrows():
                point = (row.loc[risk_measure] * 100, row.loc["return"] * notional)

                annotation = ""
                weights_dict, cash = row["optimal portfolio"]
                for ticker, weight in weights_dict.items():
                    if weight > 5e-2 or weight < -5e-2:
                        annotation += ticker + f": {weight: .2f}\n"

                annotation += f"cash: {cash: .2f}"
                annotation = annotation.rstrip("\n")

                plt.annotate(
                    annotation,
                    xy=point,
                    ha="left",
                    xytext=offset_list[row_idx],
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="#e8dff5", edgecolor="black"
                    ),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0.3", color="black"
                    ),
                )

    # create line for efficient frontier
    sns.lineplot(
        data=result_dataframe_scaled,
        x=f"{risk_measure}_percent",
        y="return_scaled",
        linewidth=3,
        zorder=1,
        label="Optimal Portfolios",
    )
    plt.legend()

    custom_portfolio_markers = ["s", "^", "v", "<", ">", "p", "h"]
    if not custom_portfolios.empty:
        for i in range(0, len(custom_portfolios)):
            portfolio = custom_portfolios.iloc[i]
            annotation = portfolio["portfolio_name"]
            plt.scatter(
                x=portfolio[risk_measure] * 100,  # Convert to percentage
                y=portfolio["return"] * notional,  # Scale by notional
                marker=custom_portfolio_markers[i],
                color=".2",
                zorder=4,
                label=annotation,
            )
    plt.legend()

    # scatter plot the single asset portfolios
    single_asset_scaled = single_asset_portfolio.copy()
    single_asset_scaled[f"{risk_measure}_percent"] = (
        single_asset_scaled[risk_measure] * 100
    )
    single_asset_scaled["return_scaled"] = single_asset_scaled["return"] * notional

    sns.scatterplot(
        data=single_asset_scaled,
        x=f"{risk_measure}_percent",
        y="return_scaled",
        hue="variance",
        size="variance",
        palette="icefire",
        legend=False,
        zorder=3,
    )

    for i in range(0, len(single_asset_portfolio)):
        plt.annotate(
            f"{single_asset_portfolio.index[i]}",
            (
                single_asset_portfolio[risk_measure][i] * 100,
                single_asset_portfolio["return"][i] * notional,
            ),
            textcoords="offset points",
            xytext=(2, 3) if i % 2 == 0 else (-4, -6),
            fontsize=7,
            ha="center",
        )

    # Set axis labels with proper scaling
    plt.xlabel("Conditional Value at Risk (CVaR %)", fontsize=10)
    plt.ylabel(f"Expected Return (${notional / 1e6:.0f}M Notional)", fontsize=10)

    if not title:
        plt.title(
            f"Efficient Frontier with {len(single_asset_portfolio)} Stocks",
            fontsize=11,
            pad=15,
        )
    else:
        plt.title(title, fontsize=11, pad=15)
    if EF_plot_png_name:
        plt.savefig(EF_plot_png_name)
    if show_plot:
        plt.show()


def get_portfolio(result, portfolio_name):
    """Extract specific portfolio from optimization results."""
    portfolio_name = portfolio_name.lower()
    if portfolio_name == "min_var":
        min_value = result["risk"].min()
        idx = result[result["risk"] == min_value].index[0]
    elif portfolio_name == "max_sharpe":
        max_sharpe = result["sharpe"].max()
        idx = result[result["sharpe"] == max_sharpe].index[0]
    elif portfolio_name == "max_return":
        max_return = result["return"].max()
        idx = result[result["return"] == max_return].index[-1]
    else:
        raise ValueError(
            "portfolio_name should be a string (e.g. min_var, max_sharpe, max_return)"
        )

    return idx


def portfolio_plot_with_backtest(
    portfolio,
    backtester,
    cut_off_date,
    backtest_plot_title,
    save_plot=False,
    results_dir="results",
):
    """
    Create side-by-side portfolio allocation and backtest performance plots.

    Displays portfolio allocation as a horizontal bar chart alongside
    cumulative returns comparison with benchmarks.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object to display allocation for
    backtester : portfolio_backtester
        Backtester object containing test portfolio and benchmarks
    cut_off_date : str
        Date to mark with vertical line on backtest plot
    backtest_plot_title : str
        Title for the backtest plot
    save_plot : bool, default False
        Whether to save the combined plot to results directory
    results_dir : str, default "results"
        Directory path where plots will be saved
    """
    # Apply consistent styling without whitegrid for portfolio plot
    sns.set_context("paper", font_scale=0.9)

    # Create subplots with appropriate sizing for side-by-side display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    # Plot portfolio allocation
    ax1 = portfolio.plot_portfolio(ax=ax1, show_plot=False)

    # Completely reset and apply very subtle grid to portfolio plot
    ax1.grid(False)  # Turn off any existing grid first
    ax1.grid(True, axis="x", alpha=0.1, color="#E0E0E0", linestyle="-", linewidth=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#E0E0E0")
    ax1.spines["bottom"].set_color("#E0E0E0")
    ax1.set_axisbelow(True)

    # Apply whitegrid style only to backtest plot
    with plt.style.context("seaborn-v0_8-whitegrid"):
        # Plot backtest results
        _, ax2 = backtester.backtest_against_benchmarks(
            plot_returns=True,
            ax=ax2,
            cut_off_date=cut_off_date,
            title=backtest_plot_title,
            save_plot=False,
        )

    # Ensure backtest grid is subtle and consistent
    ax2.grid(True, alpha=0.1, color="#E0E0E0", linewidth=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    # Save combined plot if requested
    if save_plot:
        import os

        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename
        portfolio_name = (
            portfolio.name.replace(" ", "_").lower() if portfolio.name else "portfolio"
        )
        test_method = backtester.test_method.replace("_", "")

        filename = f"combined_{portfolio_name}_{test_method}_analysis.png"
        filepath = os.path.join(results_dir, filename)

        # Save with high quality
        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        print(f"Combined plot saved: {filepath}")

    plt.show()


def compare_results(gpu_results, cpu_results):
    """
    Compare and display results from GPU and CPU solvers in tabular format.

    Args:
        gpu_results: Results from GPU solver
        cpu_results: Results from CPU solver
    """
    print("\n" + "=" * 60)
    print("SOLVER COMPARISON")
    print("=" * 60)

    # Collect all available results
    solvers = []
    if gpu_results is not None:
        # Determine GPU solver name based on results structure or default to cuOpt
        gpu_name = "cuOpt (GPU)"  # Default name for GPU results
        solvers.append((gpu_name, gpu_results))
    if cpu_results is not None:
        solvers.append((f"{cpu_results['solver']} (CPU)", cpu_results))

    if len(solvers) == 0:
        print("No results available from any solver")
        return

    # Print header
    print(
        f"{'Solver':<15} {'Solve Time (s)':<15} {'Objective':<12} "
        f"{'Return':<10} {'CVaR':<10}"
    )
    print("-" * 65)

    # Print results for each solver
    for solver_name, results in solvers:
        solve_time = results.get("solve time", 0)
        objective = results.get("obj", 0)
        portfolio_return = results.get("return", 0)
        cvar = results.get("CVaR", 0)

        print(
            f"{solver_name:<15} {solve_time:<15.4f} {objective:<12.6f} "
            f"{portfolio_return:<10.6f} {cvar:<10.6f}"
        )

    # Calculate and display objective differences if multiple results available
    if len(solvers) > 1:
        print("\nObjective Differences:")
        for i in range(len(solvers)):
            for j in range(i + 1, len(solvers)):
                solver1_name, results1 = solvers[i]
                solver2_name, results2 = solvers[j]
                obj_diff = abs(results1.get("obj", 0) - results2.get("obj", 0))
                print(f"{solver1_name} vs {solver2_name}: {obj_diff:.8f}")

    print()  # Add blank line for better readability


def download_data(dataset_dir):
    """
    Download the data for the given dataset name.
    """

    tickers = [
        'A', 'AAPL', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN',
        'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMT', 'AMZN', 'AON', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXON', 'AXP', 'AZO',
        'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BRO', 'BSX', 'BWA', 'BXP',
        'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMI', 'CMS',
        'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CVS', 'CVX',
        'D', 'DD', 'DE', 'DECK', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
        'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG',
        'EW', 'EXC', 'EXPD', 'EXR', 'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FRT',
        'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
        'HAL', 'HAS', 'HBAN', 'HD', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
        'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
        'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR',
        'L', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS',
        'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',
        'MOS', 'MPWR', 'MRK', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
        'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
        'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OXY',
        'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU',
        'PSA', 'PTC', 'PWR', 'QCOM',
        'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
        'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYK',
        'SYY', 'T', 'TAP', 'TDY', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 'TXN',
        'TXT', 'TYL', 'UDR', 'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
        'VLO', 'VMC', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
        'WAB', 'WAT', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN',
        'XEL', 'XOM', 'YUM', 'ZBH', 'ZBRA'
    ]

    start_date = "2005-01-01"
    end_date = "2025-01-01"

    data = yf.download(tickers, start=start_date, end=end_date, timeout = 30)

    data = data['Close'].dropna(axis = 1)

    data.to_csv(dataset_dir)