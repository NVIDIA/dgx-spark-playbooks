# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.6",
#     "plotly==6.3.0",
# ]
# ///

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
#
# Author: William Benton <wbenton@nvidia.com> 

import marimo

__generated_with = "0.16.3"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Getting started with marimo notebooks

    This is a [marimo](https://marimo.io) notebook.  You can learn more about marimo notebooks [here](https://docs.marimo.io), but you only need to know a few things to get started.

    1.  Notebooks are made up of _cells_ that can contain prose, executable code, or interactive UI elements.  This cell is a prose cell.
    2.  Cells can be edited and executed.  To edit a cell, double-click until the frame around it turns green.  By default, we'll execute the whole notebook when you load it, but to explicitly execute a cell (whether you've edited it or not), select it and press Shift+Enter.  This will execute the cell, record its output, and advance to the next cell.  
    3.  Unlike Jupyter notebooks, which you may have used before, marimo notebooks are _reactive_, meaning that changing the code in a cell will cause any cell depending on the changed cell's outputs to re-run.  This makes it possible to develop interactive apps and dashboards, but it also limits a potential reproducibility challenge that can come up while editing non-reactive notebooks.

    The next cell is a code cell.  It should have run automatically, but try executing it if not.  You'll be editing and re-running this cell in the next step.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default='notebook'

    x_size = 192
    y_size = 108
    feature_dims = 3

    r_mult = 0.33
    g_mult = 0.33
    b_mult = 1.0

    random_image = np.random.random(size=(y_size, x_size, feature_dims)) * np.array([r_mult, g_mult, b_mult])

    px.imshow(random_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Change the code above to show a purple-tinted image instead of a blue-tinted one.  (If you don't know exactly what it's doing, try changing some of the variables to see how it affects the picture!)

    In the notebooks we'll be working on today, there will be many opportunities to change cells and see how small code changes influence the results and performance of our models.  You should always feel free to edit notebook cells and experiment with changes to code.  We'll call out several places where we've made it especially easy to try new things (or where we've given you exercises to try) with a checkmark emoji, like this: ✅

    You just did that, so you'll notice it next time!

    # Recovering from mistakes

    Next up, we'll see how to backtrack.  Say you make a mistake in your notebook and overwrite a variable declaration that you need, or perhaps you get stuck and can't figure out how to get back to a clean slate.  Run the next cell.
    """
    )
    return


@app.cell
def _():
    def foo(x):
        """ paradoxically, this is an unhelpful function """
        return "sorry, I don't know anything about %r" %  x

    help = foo

    help(foo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Oh,  no!  We've overwritten the definition of the `help` built-in function.  This is an important way to access documentation as we're experimenting with new Python libraries.  Fortunately, we can use this as an opportunity to learn how to clean up after our mistake.

    ✅ First, edit the previous cell, deleting the line `help = foo`.  

    Then, re-run the cell.  How does it change?

    # Cleaning up

    Since some of the code we'll execute will allocate memory on the GPU, we may need to clean up after it before moving on to other notebooks.  When you're done with a notebook, simply go to the drop-down menu and select `Kernel -> Shut Down Kernel` before proceeding to the next notebook. 

    # These notebooks

    In these notebooks, you'll be using JAX to accelerate prototype implementations of a machine learning technique.  Here's how to proceed:

    * Start with [an introduction to JAX](/?file=jax-intro.py).
    * Review our basic technique by studying [an implementation of self-organizing maps in numpy](/?file=numpy-som.py).
    * Conclude with [two accelerated implementations of self-organizing maps in JAX](/?file=som-jax.py).
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
