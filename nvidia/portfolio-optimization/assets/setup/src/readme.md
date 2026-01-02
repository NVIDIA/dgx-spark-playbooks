# GPU-Accelerated Portfolio Optimization

The NVIDIA Quantitative Portfolio Optimization developer example uses NVIDIA cuOpt and CUDA-X data science libraries to transform portfolio optimization from a slow, batch process into a fast, iterative workflow.  GPU-accelerated portfolio optimization pipeline enables scalable strategy backtesting and interactive analysis.

## Overview

This package provides a comprehensive suite of tools for quantitative portfolio management, including risk-aware optimization, backtesting, and dynamic rebalancing. The library leverages GPU acceleration through NVIDIA's cuOpt solver to handle large-scale portfolio optimization problems efficiently.

## Key Features

- **GPU-Accelerated CVaR Optimization**: Utilize NVIDIA cuOpt for fast, scalable portfolio optimization
- **Multiple Modeling APIs**: Compatible with CVXPY (CPU and GPU) and cuOpt Python API (GPU)
- **Advanced Risk Management**: CVaR-based downside risk control with customizable constraints
- **Dynamic Rebalancing**: Systematic portfolio rebalancing with configurable trigger conditions
- **Comprehensive Backtesting**: Performance evaluation against benchmarks with multiple metrics
- **Scenario Generation**: Synthetic data generation using Geometric Brownian Motion
- **Flexible Constraints**: Weight bounds, leverage limits, turnover restrictions, cardinality constraints

## Module Structure

### Core Optimization

#### `cvar_optimizer.CVaR`
Main CVaR portfolio optimizer class supporting Mean-CVaR optimization with multiple solver interfaces.

**Key capabilities:**
- CVXPY solver integration (CPU)
- cuOpt solver integration (GPU)
- Customizable constraint framework
- Support for weight bounds, leverage limits, CVaR hard limits, turnover restrictions, and cardinality constraints

#### `base_optimizer.BaseOptimizer`
Abstract base class providing common functionality for optimization algorithms including weight constraint handling and portfolio state management.

#### `cvar_parameters.CvarParameters`
Configuration class for CVaR optimization parameters, constraints, and solver settings.

#### `cvar_data.CvarData`
Data container for return scenarios, asset information, and optimization inputs.

#### `cvar_utils`
Utility functions for CVaR calculations, portfolio evaluation and visualization, and optimization solver benchmark helper methods.

### Portfolio Management

#### `portfolio.Portfolio`
Portfolio class for managing asset allocations, cash holdings, and portfolio analysis.

**Features:**
- Weight and cash management
- Self-financing constraint validation
- Portfolio visualization
- Performance metrics calculation
- JSON serialization support

### Performance Analysis

#### `backtest.portfolio_backtester`
Backtesting framework for evaluating portfolio strategies against historical data and benchmarks.

**Supported methods:**
- Historical data backtesting
- KDE (Kernel Density Estimation) simulation
- Gaussian simulation

**Metrics:**
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Cumulative returns
- Volatility measures

#### `rebalance.rebalance_portfolio`
Dynamic portfolio rebalancing system with CVaR optimization and configurable trigger conditions.

**Rebalancing triggers:**
- Portfolio drift thresholds
- Performance percentage changes
- Maximum drawdown limits

**Features:**
- Rolling CVaR optimization
- Transaction cost modeling
- Performance visualization
- Baseline comparison

### Data Generation

#### `scenario_generation.ForwardPathSimulator`
Synthetic financial data generation using stochastic processes.

**Methods:**
- Geometric Brownian Motion (log_gbm)
- Path simulation for forward-looking scenarios
- Calibration from historical data

### Utilities

#### `utils`
General-purpose utilities for data processing and portfolio calculations.

**Key functions:**
- `get_input_data()`: Multi-format data loading (CSV, Parquet, Excel, JSON)
- `calculate_returns()`: Return calculation with log/linear transformations
- `calculate_log_returns()`: Log return computation
- Performance metrics and visualization helpers

## Installation

For installation instructions and prerequisites, please refer to the [main README](../README.md).

## Quick Start

### Basic Mean-CVaR Optimization

#### CVXPY

```python
from src import CvarData, CvarParameters
from src.cvar_optimizer import CVaR
import cvxpy as cp

# Load and prepare return data
returns_dict = {
    'returns': returns_data,  # Historical return scenarios
    'tickers': ['AAPL', 'MSFT', 'GOOGL'],
    'mean': mean_returns,
    'covariance': cov_matrix
}

# Configure optimization parameters
cvar_params = CvarParameters(
    alpha=0.95,                    # CVaR confidence level
    risk_aversion=1.0,             # Risk-return tradeoff
    weight_lower_bound=0.0,        # Min weight per asset
    weight_upper_bound=0.3,        # Max weight per asset
    leverage=1.0                   # No leverage
)

# Create optimizer and solve
optimizer = CVaR(returns_dict, cvar_params)
result, portfolio = optimizer.solve_optimization_problem(
    {"solver": cp.CUOPT}
) #can replace with other CPU solvers
```


#### cuOpt Python API 

```python
# Use cuOpt for GPU acceleration
api_settings = {"api": "cuopt_python"}
optimizer = CVaR(returns_dict, cvar_params, api_settings=api_settings)
result, portfolio = optimizer.solve_optimization_problem({
    "time_limit": 60
})
```

### Backtesting

```python
from src.backtest import portfolio_backtester

# Initialize backtester
backtester = portfolio_backtester(
    test_portfolio=portfolio,
    returns_dict=returns_dict,
    risk_free_rate=0.02,
    test_method="historical"
)

# Run backtest
metrics = backtester.backtest()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### Dynamic Rebalancing

```python
from src.rebalance import rebalance_portfolio

# Configure rebalancing strategy
rebalancer = rebalance_portfolio(
    dataset_directory="data/prices.csv",
    trading_start="2023-01-01",
    trading_end="2024-01-01",
    look_back_window=252,
    look_forward_window=21,
    cvar_params=cvar_params,
    solver_settings={"solver": cp.CLARABEL},
    re_optimize_criteria={
        "type": "drift",
        "threshold": 0.05
    },
    return_type="LOG"
)

# Execute rebalancing strategy
results = rebalancer.rebalance()
```

## Performance Considerations

- **GPU Acceleration**: For portfolios with 100+ assets or 5000+ scenarios, cuOpt can provide 10-100x speedup over CPU solvers
- **Constraint Handling**: Using parameter-based constraints in CVXPY can improve warm-start performance
- **Memory Management**: Large scenario sets may require chunking for GPU memory constraints


## References

For detailed API documentation and advanced usage examples, refer to the jupyter notebooks in the [`notebooks/`](../notebooks/) directory.

