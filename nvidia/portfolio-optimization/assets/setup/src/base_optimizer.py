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
Base optimization classes and utilities for portfolio optimization.

Provides abstract base classes and common functionality shared across
different optimization algorithms, including weight constraint handling
and portfolio state management.
"""

import numpy as np


class BaseOptimizer:
    """
    Base class for portfolio optimization algorithms.

    Provides common functionality for different optimization methods including
    weight constraint handling and portfolio state management.

    Attributes
    ----------
    returns_dict : dict
        Dictionary containing return data and asset information
    tickers : list
        Asset ticker symbols
    n_assets : int
        Number of assets in the portfolio
    risk_measure : str
        Risk measure type (e.g., "CVaR", "variance")
    weights_previous : np.ndarray
        Previous portfolio weights for turnover calculations
    """

    def __init__(self, returns_dict, weights_previous, risk_measure):
        """
        Initialize base optimizer with return data and portfolio state.

        Parameters
        ----------
        returns_dict : dict
            Dictionary containing asset returns data and tickers
        weights_previous : array-like or None
            Previous portfolio weights. If None or empty, creates uniform weights
        risk_measure : str
            Risk measure identifier (e.g., "CVaR", "variance")
        """
        self.returns_dict = returns_dict
        self.tickers = returns_dict["tickers"]
        self.n_assets = len(self.tickers)
        self.risk_measure = risk_measure

        if not weights_previous:  # (n_assets,) array of existing portfolio weights;
            # create uniform distributed weights if weights_previous not exist
            self.weights_previous = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights_previous = weights_previous

    def _update_weight_constraints(self, weight_constraints):
        """
        Convert weight constraints to numpy array format.

        Handles multiple input formats for weight constraints:
        - numpy array: used directly
        - dict: maps ticker names to constraint values
        - float: uniform constraint for all assets

        Parameters
        ----------
        weight_constraints : np.ndarray, dict, or float
            Weight constraint specification in various formats

        Returns
        -------
        np.ndarray
            Weight constraints as numpy array (length n_assets)

        Raises
        ------
        ValueError
            If constraint format is invalid or missing ticker specifications
        """

        # if numpy array, then use the array
        if isinstance(weight_constraints, np.ndarray):
            updated_weight_constraints = weight_constraints

        # if dict, then convert to numpy array based on the tickers
        elif isinstance(weight_constraints, dict):
            updated_weight_constraints = np.zeros(self.n_assets)
            for ticker_idx, ticker in enumerate(self.tickers):
                if ticker in weight_constraints.keys():
                    updated_weight_constraints[ticker_idx] = weight_constraints[ticker]
                elif "others" in weight_constraints.keys():
                    updated_weight_constraints[ticker_idx] = weight_constraints[
                        "others"
                    ]
                else:
                    raise ValueError(
                        "Must specify a weight constraint for each ticker or 'others'"
                    )

        # if float, then create a numpy array with the same bound for all assets
        elif isinstance(weight_constraints, float):
            updated_weight_constraints = np.full(self.n_assets, weight_constraints)
        else:
            raise ValueError("Invalid weight constraints")

        return updated_weight_constraints
