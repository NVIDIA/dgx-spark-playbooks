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

import copy
import os
import pickle
import time
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from cuopt.linear_programming.problem import (
    CONTINUOUS,
    INTEGER,
    MAXIMIZE,
    MINIMIZE,
    Problem,
)
from cuopt.linear_programming.solver_settings import SolverSettings

from . import base_optimizer
from . import cvar_utils
from .cvar_parameters import CvarParameters
from .portfolio import Portfolio

"""
Module: CVaR Optimization
=========================
This module implements data structures and a class for Conditional
Value‑at‑Risk (CVaR) portfolio optimization.
A CVaR optimizer chooses asset weights that control downside risk while
maximizing expected return (or, equivalently, minimizing a
risk‑penalised loss).

Key features
------------
* Set up problem using different interfaces (CVXPY with bounds/parameters, cuOpt).
* Build models with customizable constraints based on CvarParameter.
* Print optimization results with detailed performance metrics and allocation.

Public classes
--------------
``CVaR``
    Main CVaR portfolio optimizer class that supports multiple solver interfaces
    (CVXPY and cuOpt). Handles Mean-CVaR optimization with customizable constraints
    including weight bounds, cash allocation, leverage limits, CVaR hard limits,
    turnover restrictions, and cardinality constraints.

Usage Examples
--------------
Standard CVXPY solver (uses bounds by default):
    >>> optimizer = CVaR(returns_dict, cvar_params)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )

cuOpt GPU solver:
    >>> api_settings = {"api": "cuopt_python"}
    >>> optimizer = CVaR(returns_dict, cvar_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem({
    ...     "time_limit": 60
    ... })

CVXPY with parameters:
    >>> api_settings = {
    ...     "api": "cvxpy",
    ...     "weight_constraints_type": "parameter",
    ...     "cash_constraints_type": "parameter"
    ... }
    >>> optimizer = CVaR(returns_dict, cvar_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )

CVXPY with pickle save enabled:
    >>> api_settings = {
    ...     "api": "cvxpy",
    ...     "pickle_save_path": "cvar_problems/sp500_num-scen10000_problem.pkl"
    ... }
    >>> optimizer = CVaR(returns_dict, cvar_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )
    # Problem automatically saved during setup to:
    # cvar_problems/sp500_num-scen10000_problem.pkl
"""


class CVaR(base_optimizer.BaseOptimizer):
    """
    CVaR portfolio optimizer with multiple API support.
    Solves Mean-CVaR optimization problems with the following constraints:
        - Weight bounds
        - Cash bounds
        - Leverage constraint
        - Hard CVaR limit (optional)
        - Turnover constraint (optional)
        - Cardinality constraint (optional)

    Key features:
    - Risk-adjusted return optimization
    - Supports both CVXPY and cuOpt Python APIs
    - GPU acceleration available via cuOpt
    - Performance monitoring with timing metrics
    - Automatic setup based on API choice
    """

    def __init__(
        self,
        returns_dict: dict,
        cvar_params: CvarParameters,
        api_settings: dict = None,
        existing_portfolio: Optional[Portfolio] = None,
    ):
        """Initialize CVaR optimizer with data and constraints.

        Parameters
        ----------
        returns_dict: dict
            Input data containing regime info and CvarData instance.
        cvar_params: CvarParameters
            Constraint parameters and optimization settings (deep-copied).
        api_settings: dict, default None
            API configuration dictionary. If None, defaults to CVXPY with bounds.
            Structure: {
                'api': str,  # "cvxpy" or "cuopt_python"
                'weight_constraints_type': str,  # "parameter" or "bounds" (CVXPY only)
                'cash_constraints_type': str,   # "parameter" or "bounds" (CVXPY only)
                'pickle_save_path': str, optional  # Path to save CVXPY problem
            }
        existing_portfolio: Portfolio, optional
            An existing portfolio to measure the turnover from.
        """
        super().__init__(returns_dict, existing_portfolio, "CVaR")

        # Set default api_settings if not provided
        if api_settings is None:
            api_settings = {
                "api": "cvxpy",
                "weight_constraints_type": "bounds",
                "cash_constraints_type": "bounds",
            }

        # Validate and store API settings
        self._validate_api_settings(api_settings)
        self.api_settings = api_settings
        self.api_choice = api_settings["api"]

        self.regime_name = returns_dict["regime"]["name"]
        self.regime_range = returns_dict["regime"]["range"]
        self.data = returns_dict["cvar_data"]
        self.covariance = returns_dict["covariance"]
        self.existing_portfolio = existing_portfolio
        self.params = self._store_cvar_params(cvar_params)

        # Set up the optimization problem based on API choice
        self._setup_optimization_problem()

        self.optimal_portfolio = None

        self._result_columns = [
            "regime",
            "solver",
            "solve time",
            "return",
            "CVaR",
            "obj",
        ]

    def _store_cvar_params(self, cvar_params: CvarParameters):
        """
        Store the CVaR parameters in the optimizer.

        If w_min and w_max are input as floats, convert them to ndarrays
        with the same value repeated for all assets. Otherwise, store
        the ndarrays as is in the deepcopy.
        """
        params_copy = copy.deepcopy(cvar_params)

        params_copy.w_min = self._update_weight_constraints(params_copy.w_min)
        params_copy.w_max = self._update_weight_constraints(params_copy.w_max)

        return params_copy

    def _validate_api_settings(self, api_settings: dict):
        """
        Validate the API settings dictionary.

        Parameters
        ----------
        api_settings: dict
            API configuration dictionary to validate

        Raises
        ------
        ValueError
            If api_settings structure is invalid
        """
        if not isinstance(api_settings, dict):
            raise ValueError("api_settings must be a dictionary")

        # Validate API choice
        valid_apis = ["cvxpy", "cuopt_python"]
        api = api_settings.get("api")
        if api not in valid_apis:
            raise ValueError(f"Invalid API '{api}'. Must be one of {valid_apis}")

        # Validate constraint types (only for CVXPY)
        if api == "cvxpy":
            valid_constraint_types = ["parameter", "bounds"]

            weight_type = api_settings.get("weight_constraints_type", "bounds")
            if weight_type not in valid_constraint_types:
                raise ValueError(
                    f"Invalid weight_constraints_type '{weight_type}'. "
                    f"Must be one of {valid_constraint_types}"
                )

            cash_type = api_settings.get("cash_constraints_type", "bounds")
            if cash_type not in valid_constraint_types:
                raise ValueError(
                    f"Invalid cash_constraints_type '{cash_type}'. "
                    f"Must be one of {valid_constraint_types}"
                )

            # Validate pickle_save_path if provided
            pickle_path = api_settings.get("pickle_save_path")
            if pickle_path is not None and not isinstance(pickle_path, str):
                raise ValueError("pickle_save_path must be a string if provided")

            # Set defaults if not provided
            api_settings.setdefault("weight_constraints_type", "bounds")
            api_settings.setdefault("cash_constraints_type", "bounds")

    def _setup_optimization_problem(self):
        """
        Set up the optimization problem based on the selected API choice.

        This unified method handles setup for both CVXPY and cuOpt APIs:
        - Times the setup process
        - Scales risk aversion parameter
        - Calls the appropriate API-specific setup method
        """
        set_up_start = time.time()  # Record setup start time
        self._scale_risk_aversion()  # Adjust risk aversion parameter

        # Call the appropriate setup method based on API choice
        if self.api_choice == "cvxpy":
            self._setup_cvxpy_problem()
            self._assign_cvxpy_parameter_values()

            # Save problem to pickle if requested
            pickle_path = self.api_settings.get("pickle_save_path")
            if pickle_path is not None:
                self._save_problem_pickle(pickle_path)

        elif self.api_choice == "cuopt_python":
            (
                self._cuopt_problem,
                self._cuopt_variables,
                self.cuopt_timing_dict,
            ) = self._setup_cuopt_problem()
        else:
            # This should never happen due to validation, but add for safety
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        set_up_end = time.time()
        self.set_up_time = set_up_end - set_up_start

    def _scale_risk_aversion(self):
        """
        heuristically scale risk aversion parameter by the ratio of
        the maximum of the return over CVaR for single-asset portfolios.
        """
        single_portfolio_performance = cvar_utils.evaluate_single_asset_portfolios(self)
        self._risk_aversion_scalar = (
            single_portfolio_performance["return"]
            / single_portfolio_performance["CVaR"]
        ).max()

        self.params.update_risk_aversion(
            self.params.risk_aversion * self._risk_aversion_scalar
        )

    def _setup_cvxpy_problem(self):
        """
        Build the cvar optimization problem using natural math languages in the
        cvxpy format

        Supports the following types of problems:
            1. (LP) 'basic cvar': basic mean-cvar problem
                Minimize: lambda_risk(t + 1/(1- confidence) p^T u) - mu^T w
                Subject to: u + t >= - R^T w,
                            u >= 0,
                            sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar.

            2. (LP) 'cvar with limit': hard limit on CVaR
                Maximize: mu^T w
                Subject to: t + 1/(1- confidence) p^T u <= cvar_limit
                            u + t >= - R^T w,
                            u >= 0,
                            sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar.

            3. (LP) 'cvar with turnover': basic mean-cvar problem with an
               additional constraint on turnover (weights changed from an
               existing portfolio)
                Minimize: lambda_risk(t + 1/(1- confidence) p^T u) - mu^T w
                Subject to: u + t >= - R^T w,
                            u >= 0,
                            sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar,
                            ||w - existing_portfolio||_1 <= T_tar.

            4. (LP) 'cvar with limit and turnover':
                Maximize: mu^T w
                Subject to: t + 1/(1- confidence) p^T u <= cvar_limit
                            u + t >= - R^T w,
                            u >= 0,
                            sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar,
                            ||w - existing_ptf||_1 <= T_tar.

            5. (MILP) 'cvar with cardinality': basic mean-cvar problem with an
               additional constraint on the number of assets to be selected
                Minimize: lambda_risk(t + 1/(1- confidence) p^T u) - mu^T w
                Subject to: u + t >= - R^T w,
                            u >= 0,
                            sum{w} + c = 1,
                            w_min_i * y_i <= w_i <= w_max_i * y_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar
                            sum{y_i} <= cardinality.
                Note: y_i is a binary variable that indicates whether the
                      i-th asset is selected.

        We can also combine the above constraints to form a more complex problem.
        """

        num_assets = self.n_assets
        num_scen = len(self.data.p)

        # Create variables based on constraint type settings
        if self.api_settings["weight_constraints_type"] == "bounds":
            # Use variable bounds for weight constraints
            self.w = cp.Variable(
                num_assets,
                name="weights",
                bounds=[self.params.w_min, self.params.w_max],
            )
        else:
            # Use parameters for weight constraints (default)
            self.w = cp.Variable(num_assets, name="weights")
            self.w_min_param = cp.Parameter(num_assets, name="w_min")
            self.w_max_param = cp.Parameter(num_assets, name="w_max")

        if self.api_settings["cash_constraints_type"] == "bounds":
            # Use variable bounds for cash constraints
            self.c = cp.Variable(
                1, name="cash", bounds=[self.params.c_min, self.params.c_max]
            )
        else:
            # Use parameters for cash constraints (default)
            self.c = cp.Variable(1, name="cash")
            self.c_min_param = cp.Parameter(name="c_min")
            self.c_max_param = cp.Parameter(name="c_max")

        # Create other auxiliary variables
        u = cp.Variable(num_scen, nonneg=True)
        t = cp.Variable(1)

        # Create parameters for optimization parameters (always parameters)
        self.risk_aversion_param = cp.Parameter(nonneg=True, name="risk_aversion")
        self.L_tar_param = cp.Parameter(nonneg=True, name="L_tar")
        self.T_tar_param = cp.Parameter(nonneg=True, name="T_tar")
        self.cvar_limit_param = cp.Parameter(nonneg=True, name="cvar_limit")
        self.cardinality_param = cp.Parameter(name="cardinality")

        # set up expressions used in the optimization process
        self.expected_ptf_returns = self.data.mean.T @ self.w
        self.cvar_risk = t + 1 / (1 - self.params.confidence) * self.data.p @ u
        scenario_ptf_returns = self.data.R.T @ self.w

        # Add variable bounds constraints (only if using parameter constraints)
        constraints = []
        if self.api_settings["weight_constraints_type"] == "parameter":
            constraints.extend(
                [
                    self.w_min_param <= self.w,
                    self.w <= self.w_max_param,
                ]
            )
        if self.api_settings["cash_constraints_type"] == "parameter":
            constraints.extend(
                [
                    self.c_min_param <= self.c,
                    self.c <= self.c_max_param,
                ]
            )

        # set up the common constraints shared across all problem types
        if self.params.cardinality is not None:
            self._problem_type = "MILP"
            print(f"{'=' * 50}")
            print("MIXED-INTEGER LINEAR PROGRAMMING (MILP) SETUP")
            print(f"{'=' * 50}")
            print(f"Cardinality Constraint: K ≤ {self.params.cardinality} assets")
            print(f"{'=' * 50}")
            y = cp.Variable(num_assets, boolean=True, name="cardinality")

            # Handle cardinality constraints based on weight constraint type
            if self.api_settings["weight_constraints_type"] == "parameter":
                constraints.extend(
                    [
                        cp.multiply(self.w_min_param, y) <= self.w,
                        self.w <= cp.multiply(self.w_max_param, y),
                    ]
                )
            else:
                # For bounds-based constraints, we need to add explicit
                # cardinality constraints
                constraints.extend(
                    [
                        cp.multiply(self.params.w_min, y) <= self.w,
                        self.w <= cp.multiply(self.params.w_max, y),
                    ]
                )

            constraints.extend(
                [
                    cp.sum(self.w) + self.c == 1,
                    u + t + scenario_ptf_returns >= 0,
                    cp.norm1(self.w) <= self.L_tar_param,
                    cp.sum(y) <= self.cardinality_param,
                ]
            )
        else:
            constraints.extend(
                [
                    u + t + scenario_ptf_returns >= 0,
                    cp.sum(self.w) + self.c == 1,
                    cp.norm1(self.w) <= self.L_tar_param,
                ]
            )

        # set up objective
        if self.params.cvar_limit is None:
            obj = cp.Minimize(
                self.risk_aversion_param * self.cvar_risk - self.expected_ptf_returns
            )
        else:
            obj = cp.Maximize(self.expected_ptf_returns)
            constraints.append(self.cvar_risk <= self.cvar_limit_param)

        # set up turnover constraint
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            w_prev = np.array(self.existing_portfolio.weights)
            z = self.w - w_prev
            constraints.append(cp.norm(z, 1) <= self.T_tar_param)

        # set up group constraints
        if self.params.group_constraints is not None:
            for group_constraint in self.params.group_constraints:
                tickers_index = [
                    self.tickers.index(ticker) for ticker in group_constraint["tickers"]
                ]
                constraints.append(
                    cp.sum(self.w[tickers_index])
                    <= group_constraint["weight_bounds"]["w_max"]
                )
                constraints.append(
                    cp.sum(self.w[tickers_index])
                    >= group_constraint["weight_bounds"]["w_min"]
                )

        # store the optimization problem
        self.optimization_problem = cp.Problem(obj, constraints)

    def _assign_cvxpy_parameter_values(self):
        """
        Assign values to all CVXPY parameters from current data and parameter settings.

        This function should be called after the CVXPY problem is set up and whenever
        parameter values need to be updated without rebuilding the entire problem.
        """
        # Assign basic constraint parameters (only if they exist as parameters)
        if self.api_settings["weight_constraints_type"] == "parameter":
            self.w_min_param.value = self.params.w_min
            self.w_max_param.value = self.params.w_max

        if self.api_settings["cash_constraints_type"] == "parameter":
            self.c_min_param.value = self.params.c_min
            self.c_max_param.value = self.params.c_max

        # Assign optimization parameters (always parameters)
        self.risk_aversion_param.value = self.params.risk_aversion
        self.L_tar_param.value = self.params.L_tar

        # Assign optional parameters
        if self.params.T_tar is not None:
            self.T_tar_param.value = self.params.T_tar

        if self.params.cvar_limit is not None:
            self.cvar_limit_param.value = self.params.cvar_limit

        if self.params.cardinality is not None:
            self.cardinality_param.value = self.params.cardinality

    def _setup_cuopt_problem(self):
        """
        Set up CVaR optimization problem using cuOpt Python API.

        Creates cuOpt Problem instance with variables, constraints, and objective
        for CVaR portfolio optimization. Note that cuOpt does not support
        vectorized variables, so all variables and constraints are set up using loops.

        Currently supports:
        - Weight bounds and cash constraints
        - Leverage constraints
        - Turnover constraints
        - CVaR hard limits
        - Cardinality constraints
        - Group constraints

        Returns
        -------
        problem : cuopt.linear_programming.problem.Problem
            cuOpt problem instance ready to solve
        variables : dict
            Dictionary containing problem variables for result extraction
        timing_dict : dict
            Timing information for each setup loop in seconds
        """
        num_assets = self.n_assets
        num_scen = len(self.data.p)

        # Initialize timing dictionary
        timing_dict = {}

        # Create a new cuOpt problem
        start_time = time.time()
        problem = Problem("CVaR Portfolio Optimization")
        timing_dict["problem_creation"] = time.time() - start_time

        # Initialize variable storage
        variables = {}

        # Add portfolio weight variables (continuous)
        start_time = time.time()
        variables["w"] = []
        for i in range(num_assets):
            w_var = problem.addVariable(
                lb=float(self.params.w_min[i]),
                ub=float(self.params.w_max[i]),
                vtype=CONTINUOUS,
                name=f"w_{i}",
            )
            variables["w"].append(w_var)
        timing_dict["weight_variables"] = time.time() - start_time

        # Add cash variable
        start_time = time.time()
        variables["c"] = problem.addVariable(
            lb=float(self.params.c_min),
            ub=float(self.params.c_max),
            vtype=CONTINUOUS,
            name="cash",
        )
        timing_dict["cash_variable"] = time.time() - start_time

        # Add auxiliary variables for CVaR calculation
        start_time = time.time()
        variables["u"] = []
        for j in range(num_scen):
            u_var = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name=f"u_{j}")
            variables["u"].append(u_var)
        timing_dict["auxiliary_variables"] = time.time() - start_time

        # Add CVaR threshold variable
        start_time = time.time()
        variables["t"] = problem.addVariable(vtype=CONTINUOUS, name="t")
        timing_dict["threshold_variable"] = time.time() - start_time

        # Add budget constraint: sum(w) + c = 1
        start_time = time.time()
        budget_expr = variables["c"]
        for i in range(num_assets):
            budget_expr = budget_expr + variables["w"][i]
        problem.addConstraint(budget_expr == 1.0, name="budget_constraint")
        timing_dict["budget_constraint"] = time.time() - start_time

        # Add CVaR scenario constraints: u[j] + t >= -sum(R[i,j] * w[i])
        start_time = time.time()
        for j in range(num_scen):
            scenario_return_expr = variables["t"] + variables["u"][j]
            for i in range(num_assets):
                scenario_return_expr = (
                    scenario_return_expr + self.data.R[i, j] * variables["w"][i]
                )
            problem.addConstraint(
                scenario_return_expr >= 0.0, name=f"cvar_scenario_{j}"
            )
        timing_dict["cvar_constraints"] = time.time() - start_time

        # Add leverage constraint: sum(|w[i]|) <= L_tar
        # For cuOpt, we need to add separate variables for positive and negative parts
        if self.params.L_tar < float("inf"):
            start_time = time.time()
            leverage_expr = None
            variables["w_pos"] = []
            variables["w_neg"] = []

            for i in range(num_assets):
                w_pos = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name=f"w_pos_{i}")
                w_neg = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name=f"w_neg_{i}")
                variables["w_pos"].append(w_pos)
                variables["w_neg"].append(w_neg)

                # w[i] = w_pos[i] - w_neg[i]
                problem.addConstraint(
                    variables["w"][i] == w_pos - w_neg, name=f"weight_decomposition_{i}"
                )

                # Add to leverage sum
                if leverage_expr is None:
                    leverage_expr = w_pos + w_neg
                else:
                    leverage_expr = leverage_expr + w_pos + w_neg

            # Leverage constraint
            problem.addConstraint(
                leverage_expr <= self.params.L_tar, name="leverage_constraint"
            )
            timing_dict["leverage_constraints"] = time.time() - start_time
        else:
            timing_dict["leverage_constraints"] = 0.0

        # Add cardinality constraints (requires integer variables)
        if self.params.cardinality is not None:
            start_time = time.time()
            variables["y"] = []

            # Add binary/integer variables for asset selection (constrained to 0-1)
            for i in range(num_assets):
                y_var = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"y_{i}")
                variables["y"].append(y_var)

            # Add cardinality constraint: sum(y_i) <= cardinality
            cardinality_expr = None
            for i in range(num_assets):
                if cardinality_expr is None:
                    cardinality_expr = variables["y"][i]
                else:
                    cardinality_expr = cardinality_expr + variables["y"][i]
            problem.addConstraint(
                cardinality_expr <= self.params.cardinality,
                name="cardinality_constraint",
            )

            # Add gating constraints: w_min_i * y_i <= w_i <= w_max_i * y_i
            for i in range(num_assets):
                # Lower bound constraint: w[i] >= w_min[i] * y[i]
                problem.addConstraint(
                    variables["w"][i]
                    >= float(self.params.w_min[i]) * variables["y"][i],
                    name=f"cardinality_lower_{i}",
                )
                # Upper bound constraint: w[i] <= w_max[i] * y[i]
                problem.addConstraint(
                    variables["w"][i]
                    <= float(self.params.w_max[i]) * variables["y"][i],
                    name=f"cardinality_upper_{i}",
                )

            timing_dict["cardinality_constraints"] = time.time() - start_time
            print(
                f"Cardinality Constraint: K ≤ {self.params.cardinality} assets (MILP)"
            )
        else:
            timing_dict["cardinality_constraints"] = 0.0

        # Add turnover constraint if existing portfolio is provided
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            start_time = time.time()
            w_prev = np.array(self.existing_portfolio.weights)
            turnover_expr = None
            variables["turnover_pos"] = []
            variables["turnover_neg"] = []

            for i in range(num_assets):
                to_pos = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"turnover_pos_{i}"
                )
                to_neg = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"turnover_neg_{i}"
                )
                variables["turnover_pos"].append(to_pos)
                variables["turnover_neg"].append(to_neg)

                # (w[i] - w_prev[i]) = turnover_pos[i] - turnover_neg[i]
                problem.addConstraint(
                    variables["w"][i] - float(w_prev[i]) == to_pos - to_neg,
                    name=f"turnover_decomposition_{i}",
                )

                # Add to turnover sum
                if turnover_expr is None:
                    turnover_expr = to_pos + to_neg
                else:
                    turnover_expr = turnover_expr + to_pos + to_neg

            # Turnover constraint
            problem.addConstraint(
                turnover_expr <= self.params.T_tar, name="turnover_constraint"
            )
            timing_dict["turnover_constraints"] = time.time() - start_time
        else:
            timing_dict["turnover_constraints"] = 0.0

        # Add group constraints
        if self.params.group_constraints is not None:
            start_time = time.time()
            for group_idx, group_constraint in enumerate(self.params.group_constraints):
                # Get indices for tickers in this group
                tickers_index = [
                    self.tickers.index(ticker) for ticker in group_constraint["tickers"]
                ]

                # Build sum expression for weights in this group
                group_sum_expr = None
                for i in tickers_index:
                    if group_sum_expr is None:
                        group_sum_expr = variables["w"][i]
                    else:
                        group_sum_expr = group_sum_expr + variables["w"][i]

                # Add upper and lower bound constraints for the group
                problem.addConstraint(
                    group_sum_expr <= group_constraint["weight_bounds"]["w_max"],
                    name=f"group_{group_idx}_upper",
                )
                problem.addConstraint(
                    group_sum_expr >= group_constraint["weight_bounds"]["w_min"],
                    name=f"group_{group_idx}_lower",
                )

            timing_dict["group_constraints"] = time.time() - start_time
            print(f"Group Constraints: {len(self.params.group_constraints)} groups")
        else:
            timing_dict["group_constraints"] = 0.0

        # Set up objective function
        start_time = time.time()
        expected_return_expr = None
        for i in range(num_assets):
            term = float(self.data.mean[i]) * variables["w"][i]
            if expected_return_expr is None:
                expected_return_expr = term
            else:
                expected_return_expr = expected_return_expr + term

        cvar_expr = variables["t"]
        for j in range(num_scen):
            cvar_expr = (
                cvar_expr
                + float(self.data.p[j] / (1 - self.params.confidence))
                * variables["u"][j]
            )

        if self.params.cvar_limit is None:
            # Minimize: risk_aversion * CVaR - expected_return
            objective_expr = (
                float(self.params.risk_aversion) * cvar_expr - expected_return_expr
            )
            problem.setObjective(objective_expr, sense=MINIMIZE)
        else:
            # Maximize: expected_return subject to CVaR <= cvar_limit
            problem.setObjective(expected_return_expr, sense=MAXIMIZE)
            problem.addConstraint(
                cvar_expr <= self.params.cvar_limit, name="cvar_limit_constraint"
            )
        timing_dict["objective_setup"] = time.time() - start_time

        print(f"{'=' * 50}")
        print("cuOpt PROBLEM SETUP COMPLETED")
        print(f"{'=' * 50}")
        print(
            f"Variables: {num_assets} weights + 1 cash + {num_scen} auxiliary + "
            f"1 threshold"
        )
        if self.params.cardinality is not None:
            print(f"           + {num_assets} cardinality (integer)")
        if self.params.L_tar < float("inf"):
            print(f"           + {2 * num_assets} leverage decomposition")
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            print(f"           + {2 * num_assets} turnover decomposition")
        if self.params.group_constraints is not None:
            print(
                f"           + {len(self.params.group_constraints)} group constraints"
            )
        print(
            f"Constraints: Budget + {num_scen} CVaR scenarios + additional constraints"
        )
        if self.params.cardinality is not None:
            print("Problem Type: MILP (Mixed-Integer Linear Programming)")
        else:
            print("Problem Type: LP")
        print(f"{'=' * 50}")

        return problem, variables, timing_dict

    def _print_cuopt_timing(self, timing_dict):
        """Print detailed timing information for cuOpt problem setup loops.

        Parameters
        ----------
        timing_dict : dict
            Dictionary containing timing information for each setup phase.
        """
        print("\ncuOpt SETUP TIMING BREAKDOWN")
        print(f"{'-' * 40}")
        total_time = sum(timing_dict.values())
        for phase, time_taken in timing_dict.items():
            percentage = (time_taken / total_time * 100) if total_time > 0 else 0
            print(
                f"{phase.replace('_', ' ').title():<25}: {time_taken:.6f}s "
                f"({percentage:.1f}%)"
            )
        print(f"{'-' * 40}")
        print(f"{'Total Setup Time':<25}: {total_time:.6f}s (100.0%)")
        print()

    def _solve_cuopt_problem(self, solver_settings: dict = None):
        """
        Solve CVaR optimization using cuOpt.

        Parameters
        ----------
        solver_settings : dict, optional
            cuOpt solver configuration. If None, uses default settings.
            Example: {"time_limit": 60}

        Returns
        -------
        result_row : pd.Series
            Performance metrics: regime, solve_time, return, CVaR, objective
        weights : np.ndarray
            Optimal asset weights
        cash : float
            Optimal cash allocation
        """
        # Configure solver settings
        settings = SolverSettings()
        if solver_settings:
            for param, value in solver_settings.items():
                settings.set_parameter(param, value)

        # Solve the problem
        total_start = time.time()
        self._cuopt_problem.solve(settings)
        total_end = time.time()
        total_time = total_end - total_start
        solve_time = self._cuopt_problem.SolveTime
        self.cuopt_api_overhead = total_time - solve_time

        # Check solution status
        if self._cuopt_problem.Status.name != "Optimal":
            raise RuntimeError(
                f"cuOpt failed to find optimal solution. Status: "
                f"{self._cuopt_problem.Status.name}"
            )

        # Extract solution
        weights = np.array([var.getValue() for var in self._cuopt_variables["w"]])
        cash = self._cuopt_variables["c"].getValue()

        # Calculate performance metrics
        expected_return = np.dot(self.data.mean, weights)

        # Calculate CVaR
        t_value = self._cuopt_variables["t"].getValue()
        u_values = np.array([var.getValue() for var in self._cuopt_variables["u"]])
        cvar_value = t_value + np.dot(self.data.p, u_values) / (
            1 - self.params.confidence
        )

        objective_value = self._cuopt_problem.ObjValue

        result_row = pd.Series(
            [
                self.regime_name,
                "cuOpt",
                solve_time,
                expected_return,
                cvar_value,
                objective_value,
            ],
            index=self._result_columns,
        )

        print(f"cuOpt solution found in {solve_time:.2f} seconds")
        print(f"Status: {self._cuopt_problem.Status.name}")
        print(f"Objective value: {objective_value:.6f}")

        return result_row, weights, cash

    def _solve_cvxpy_problem(self, solver_settings: dict):
        """
        solve the optimization problem using the user-specified solver and settings

        Parameters
        ----------
        solver_settings: dict
            Solver configuration dict for CVXPY.Problem.solve().
            Example: {"solver": cp.CLARABEL, "verbose": True}

        Returns
        -------
        result_row: pd.Series
            Performance metrics: regime, solve_time, return, CVaR, objective.
        weights: np.ndarray
            Optimal asset weights.
        cash: float
            Optimal cash allocation.
        """

        self.optimization_problem.solve(**solver_settings)
        weights = self.w.value
        cash = self.c.value

        solver_stats = getattr(self.optimization_problem, "solver_stats", None)

        reported_solve_time = (
            getattr(solver_stats, "solve_time", None)
            if solver_stats is not None
            else None
        )
        if reported_solve_time is None:
            print("no reported solve time!!!")
        solver_time = (
            float(reported_solve_time)
            if reported_solve_time is not None
            else self.optimization_problem._solve_time
        )

        self.cvxpy_api_overhead = (
            self.optimization_problem._solve_time - solver_time
            if reported_solve_time is not None
            else None
        )

        result_row = pd.Series(
            [
                self.regime_name,
                str(solver_settings["solver"]),
                solver_time,
                self.expected_ptf_returns.value,
                self.cvar_risk.value[0],
                self.optimization_problem.value,
            ],
            index=self._result_columns,
        )

        return result_row, weights, cash

    def _save_problem_pickle(self, pickle_save_path: str):
        """
        Save the CVXPY optimization problem to a pickle file.

        Parameters
        ----------
        pickle_save_path : str
            Path where to save the pickle file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)

            with open(pickle_save_path, "wb") as f:
                pickle.dump(self.optimization_problem, f)
            print(f"CVaR problem saved to: {pickle_save_path}")
        except Exception as e:
            print(f"Warning: Failed to save CVaR problem to pickle: {e}")

    def _print_CVaR_results(
        self,
        result_row: pd.Series,
        portfolio: Portfolio,
        time_results: dict,
        min_percentage: float = 1,
    ):
        """
        Display CVaR optimization results and optimized portfolio allocation.

        Parameters
        ----------
        result_row : pd.Series
            optimization results
        portfolio : <portfolio object>
            portfolio to display the readable allocation
        time_results : dict
            Additional timing breakdown (e.g., data prep, post-processing).
        min_percentage : float, default 1
            Only assets with absolute allocation >= min_percentage%
            will be shown.
        """
        solver_name = result_row["solver"]
        solve_time = result_row["solve time"]
        expected_return = result_row["return"]
        cvar_value = result_row["CVaR"]
        objective_value = result_row["obj"]

        # Main header
        print(f"\n{'=' * 60}")
        print("CVaR OPTIMIZATION RESULTS")
        print(f"{'=' * 60}")

        # Problem configuration section
        print("PROBLEM CONFIGURATION")
        print(f"{'-' * 30}")
        print(f"Solver:              {solver_name}")
        print(f"Regime:              {self.regime_name}")
        print(f"Time Period:         {self.regime_range[0]} to {self.regime_range[1]}")
        print(f"Scenarios:           {len(self.data.p):,}")
        print(f"Assets:              {self.n_assets}")
        print(f"Confidence Level:    {self.params.confidence:.1%}")

        if self.params.cardinality is not None:
            print(f"Cardinality Limit:   {self.params.cardinality} assets")
        if self.params.cvar_limit is not None:
            print(f"CVaR Hard Limit:     {self.params.cvar_limit:.4f}")
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            print(f"Turnover Constraint: {self.params.T_tar:.3f}")

        # Performance metrics section
        print("\nPERFORMANCE METRICS")
        print(f"{'-' * 30}")
        print(
            f"Expected Return:     {expected_return:.6f} ({expected_return * 100:.4f}%)"
        )
        print(
            f"CVaR ({self.params.confidence:.0%}):          {cvar_value:.6f} "
            f"({cvar_value * 100:.4f}%)"
        )
        print(f"Objective Value:     {objective_value:.6f}")

        # Timing section
        print("\nSOLVING PERFORMANCE")
        print(f"{'-' * 30}")
        # Print setup time based on solver type
        if hasattr(self, "set_up_time"):
            print(f"Setup Time:          {self.set_up_time:.4f} seconds")
        if hasattr(self, "cvxpy_api_overhead"):
            print(f"CVXPY API Overhead:  {self.cvxpy_api_overhead:.4f} seconds")
        if hasattr(self, "cuopt_api_overhead"):
            print(f"cuOpt API Overhead:  {self.cuopt_api_overhead:.4f} seconds")
        print(f"Solve Time:          {solve_time:.4f} seconds")

        for key, value in time_results.items():
            print(f"{key.title():20} {value:.4f} seconds")

        # Portfolio allocation section
        print("\nOPTIMAL PORTFOLIO ALLOCATION")
        print(f"{'-' * 30}")
        portfolio.print_clean(verbose=True, min_percentage=min_percentage)

        print(f"{'=' * 60}\n")

    def solve_optimization_problem(
        self, solver_settings: dict = None, print_results: bool = True
    ):
        """
        Unified solve method that calls the appropriate API-specific solver.

        This method automatically calls the correct solver based on the api_choice
        specified during initialization.

        Parameters
        ----------
        solver_settings : dict, optional
            Solver configuration. Format depends on API choice:
            - CVXPY: {"solver": cp.CLARABEL, "verbose": True}
            - cuOpt: {"pdlp_solver_mode": 1, "log_to_console": False}
            If None, uses default settings for the chosen API.
        print_results : bool, default True
            Enable formatted result output to console.

        Returns
        -------
        result_row : pd.Series
            Performance metrics: regime, solve_time, return, CVaR, objective.
        portfolio : Portfolio
            Optimized portfolio with weights and cash allocation.

        Raises
        ------
        ValueError
            If api_choice is not supported or required settings are missing.
        """
        time_results = {}

        # Call appropriate solve method based on API choice
        if self.api_choice == "cvxpy":
            if solver_settings is None or solver_settings.get("solver") is None:
                raise ValueError("A solver must be provided for CVXPY API")
            result_row, weights, cash = self._solve_cvxpy_problem(solver_settings)
            portfolio_name = str(solver_settings["solver"]) + "_optimal"
        elif self.api_choice == "cuopt_python":
            result_row, weights, cash = self._solve_cuopt_problem(solver_settings)
            portfolio_name = "cuOpt_optimal"
        else:
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        # Create portfolio object with results
        portfolio = Portfolio(
            name=portfolio_name,
            tickers=self.tickers,
            weights=weights,
            cash=cash,
            time_range=self.regime_range,
        )

        # Print results if requested
        if print_results:
            self._print_CVaR_results(result_row, portfolio, time_results)

        return result_row, portfolio

    def _extract_problem_cone_data(self, problem_data_dir: str):
        """
        Extract the cone data from the problem and save to pickle file.
        Parameters for benchmarking conic solvers.
        ----------
        problem_data_dir: str
            Path where to save the pickle file
        """

        data = self.optimization_problem.get_problem_data("SCS")
        P = data[0].get("P", None)
        q = data[0].get("c", None)  # CVXPy uses 'c', Clarabel uses 'q'
        A = data[0].get("A", None)
        b = data[0].get("b", None)
        dims = data[0].get("dims", None)

        # Create directory if it doesn't exist
        os.makedirs(problem_data_dir, exist_ok=True)

        # Create specific filename with problem details
        regime_name = getattr(self, "regime_name", "unknown")
        num_scenarios = getattr(self.params, "num_scen", "unknown")

        filename = f"cvar_{regime_name}_{num_scenarios}scen.pkl"
        pickle_file_path = os.path.join(problem_data_dir, filename)

        # Save the entire data object as pickle
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Problem data saved to: {pickle_file_path}")

        return P, q, A, b, dims
