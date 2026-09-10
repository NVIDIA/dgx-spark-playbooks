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

from dataclasses import dataclass

import numpy as np


@dataclass
class CvarData:
    """
    Data structure holding all scenario and statistical information required
    for CVaR optimization.

    Attributes
    ----------
    mean : np.ndarray
        Array of shape (n_assets,) of expected asset returns.
    R : np.ndarray
        Scenario deviations, shape (num_scenarios, n_assets),
        each row is asset return deviation for a scenario.
    p : np.ndarray
        Scenario probabilities, shape (num_scenarios,), summing to 1.

    Examples
    --------
    >>> import numpy as np
    >>> # Create data for 3 assets with 5 scenarios
    >>> mean = np.array([0.08, 0.10, 0.12])
    >>> R = np.array([
    ...     [-0.02, 0.01, 0.03],
    ...     [0.01, -0.01, 0.02],
    ...     [0.03, 0.02, -0.01],
    ...     [-0.01, 0.02, 0.01],
    ...     [0.02, -0.02, 0.00]
    ... ])
    >>> p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> data = CvarData(mean=mean, R=R, p=p)
    >>> print(data.mean.shape)  # (3,)
    >>> print(data.R.shape)  # (5, 3)
    >>> print(data.p.sum())  # 1.0

    """

    mean: np.ndarray
    R: np.ndarray
    p: np.ndarray
