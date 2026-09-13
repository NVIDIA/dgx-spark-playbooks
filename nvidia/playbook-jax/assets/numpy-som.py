# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.6",
#     "opencv-python==4.12.0.88",
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

__generated_with = "0.15.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.output.append(mo.md(
        r"""
    # Self-organizing maps in numpy

    The [self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map) is a classic unsupervised learning technique for dimensionality reduction.  It takes a set of high-dimensional training examples and produces a low-dimensional map, typically consisting of a grid or cube of high-dimensional vectors.  

    Informally, the training algorithm proceeds by examining each training example, identifying the most similar node to the given example in the map and influencing the neighborhood of nodes around this one to become slightly more like the training example. The size of the influenced neighborhood and the amount of influence exerted both decrease over time. (There is also a batch algorithm, which calculates map weights directly from the sets of examples that mapped to a given neighborhood with a previous version of the map.)

    The following video shows an example of training a self-organizing map from color data using a batch algorithm, from a random initial map to a relatively converged set of colorful clusters.
    """
    ))
    mo.output.append(mo.video("batch-som.mp4"))
    mo.output.append(mo.md(
        r"""

    In this notebook, we'll see how to develop two implementations of this algorithm in numpy. **If you're primarily interested in acceleration, feel free to just skim this notebook without running it!**  In future notebooks, you'll learn how to accelerate your implementation with JAX.  The implementation we'll develop is a prototype — there are lots of things you might want to improve or change before using it in a real system — but it will show how we can accelerate a relatively realistic codebase realizing a more involved ML technique.
    """
    ))
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There are several ways to initialize a self-organizing map.  Here, we'll initialize our map with random vectors.  The result of this function is a matrix with a row for every element in a self-organizing map; each row contains uniformly-sampled random numbers between 0 and 1.""")
    return


@app.cell
def _(np):
    def init_som(xdim, ydim, fdim, seed):
      rng = np.random.default_rng(seed)
      return rng.random(size=(xdim, ydim, fdim)).reshape(xdim * ydim, fdim)
    return (init_som,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It's often helpful to visualize the results of our code.  Here we'll plot an example random map of three-element vectors, interpreting each vector as a color by interpreting each feature value as a red, green, or blue intensity.  Because our code will refer to maps with x- and y-coordinates but represent them as matrices (where we'd list rows first and then columns), we will often rearrange or transpose our data (with methods like `reshape` and `swapaxes` or `T`) before plotting it.""")
    return


@app.cell
def _(init_som):
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default='notebook'

    x_size = 192
    y_size = 108
    feature_dims = 3

    random_map = init_som(x_size, y_size, feature_dims, 42)

    px.imshow(random_map.reshape(x_size, y_size, feature_dims).swapaxes(0,1))
    return px, x_size, y_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our neighborhood function describes the part of the map influenced by a given sample.  It takes two ranges (corresponding to every index along the x dimension and every index along the y dimension), the coordinates of the center of the neighborhood, and the radiuses of influence along each dimension.""")
    return


@app.cell
def _(np):
    def neighborhood(range_x, range_y, center_x, center_y, x_sigma, y_sigma):
      x_distance = np.abs(center_x - range_x)
      x_neighborhood = np.exp(- np.square(x_distance) / np.square(x_sigma))

      y_distance = np.abs(center_y - range_y)
      y_neighborhood = np.exp(- np.square(y_distance) / np.square(y_sigma))

      return np.outer(x_neighborhood, y_neighborhood)
    return (neighborhood,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see an example neighborhood in the heatmap below.""")
    return


@app.cell
def _(neighborhood, np, px, x_size, y_size):
    center_x = 12
    center_y = 48
    sigma_x = 96
    sigma_y = 54

    px.imshow(neighborhood(np.arange(x_size), np.arange(y_size), center_x, center_y, sigma_x, sigma_y).T)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here's the basic online (i.e., one sample at a time) training algorithm.""")
    return


@app.cell
def _(init_som, neighborhood, np):
    def train_som_online(examples, xdim, ydim, x_sigma, y_sigma, max_iter, seed=None, frame_callback=None):
      t = -1

      exs = examples.copy()
      fdim = exs.shape[-1]

      x_sigmas = np.linspace(x_sigma, max(2, x_sigma*.05), max_iter)
      y_sigmas = np.linspace(y_sigma, max(2, y_sigma*.05), max_iter)
      alphas = np.geomspace(0.35, 0.01, max_iter)

      range_x, range_y = np.arange(xdim), np.arange(ydim)

      hood = None
      som = init_som(xdim, ydim, fdim, seed)

      rng = np.random.default_rng(seed)
      while t < max_iter:
        rng.shuffle(exs)
        for ex in exs:
          t = t + 1
          if t == max_iter:
            break

          # best matching unit (by euclidean distance)
          bmu_idx = np.argmin(((ex - som) ** 2).sum(axis = 1))

          bmu = som[bmu_idx]

          center_x, center_y = np.divmod(bmu_idx, ydim)

          hood = neighborhood(range_x, range_y, center_x, center_y, x_sigmas[t], y_sigmas[t]).reshape(-1, 1)

          update = np.multiply(((ex - som) * alphas[t]), hood)

          frame_callback and frame_callback(t - 1, ex, hood, som)
          np.add(som, update, som)
          np.clip(som, 0, 1, som)

      frame_callback and frame_callback(t, ex, hood, som)
      return som
    return (train_som_online,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We'll now introduce a small class to track our history at each training epoch or example -- this is useful for debugging and visualizing an entire training process.""")
    return


@app.class_definition
class HistoryCallback(object):

    def __init__(self, xdim, ydim, fdim, epoch_pred):
        self.frames = dict()
        self.meta = dict()
        self.xdim = xdim
        self.ydim = ydim
        self.fdim = fdim
        self.epoch_pred = epoch_pred

    def __call__(self, epoch, ex, hood, som, **meta):
        if self.epoch_pred(epoch):
            self.frames[epoch] = (ex, hood, som.copy())
        if meta is not None:
            self.meta[epoch] = meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we'll train a small map on random color data, storing one history snapshot for every 20 examples.""")
    return


@app.cell
def _(np, train_som_online):
    fc = HistoryCallback(240,135,3, lambda x: x%20 == 0)
    color_som =\
      train_som_online(np.random.random(size=(1000, 3)), 
                       240, 135, 
                       120, 70, 
                       50000, None, fc)
    return (color_som,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Mouse over the above cell.  How long did it take to execute?

    Here's a little example function you can pass a history callback object to in order to save every updated map to a PNG file:
    """
    )
    return


@app.cell
def _(np):
    import cv2

    def plot_history(fc, plot_prefix="plot"):
        for k in fc.frames.keys():
            ex, hood, som = fc.frames[k]

            # convert image to BGR data to save with opencv
            brg = (np.roll(som, 1, axis=1) * 255).astype("uint8")
            brg = brg.reshape(fc.xdim, fc.ydim, fc.fdim).swapaxes(0, 1).reshape(fc.ydim, fc.xdim, fc.fdim)
            cv2.imwrite(f"{plot_prefix}-{k:06}.png", brg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also plot just the final map.""")
    return


@app.cell
def _(color_som, px):
    px.imshow(color_som.reshape(240,135,3).swapaxes(0,1)).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, let's consider the batch variant of the algorithm.""")
    return


@app.cell
def _(init_som, neighborhood, np):
    def train_som_batch(examples, xdim, ydim, x_sigma, y_sigma, epochs, min_sigma_frac=0.1, seed=None, frame_callback=None):
      t = 0

      exs = examples.copy()
      fdim = exs.shape[-1]

      x_sigmas = np.geomspace(x_sigma, max(2, xdim*min_sigma_frac), epochs)
      y_sigmas = np.geomspace(y_sigma, max(2, ydim*min_sigma_frac), epochs)

      range_x, range_y = np.arange(xdim), np.arange(ydim)

      hood = None
      som = init_som(xdim, ydim, fdim, seed)

      rng = np.random.default_rng(seed)
      while t < epochs:
        hoods = np.zeros((xdim*ydim, xdim * ydim, 1))
        hoods_accum = np.zeros((xdim * ydim, 1))
        updates = np.zeros((len(examples), xdim*ydim, fdim))
        for (i, ex) in enumerate(exs):

          # best matching unit (euclidean distance)

          bmu_idx = np.argmin(((ex - som) ** 2).sum(axis = 1))

          bmu = som[bmu_idx]

          # cache the neighborhood for this unit (if we haven't seen it yet)
          if np.max(hoods[bmu_idx]) == 0:
            center_x, center_y = np.divmod(bmu_idx, ydim)
            hoods[bmu_idx] = neighborhood(range_x, range_y, center_x, center_y, x_sigmas[t], y_sigmas[t]).reshape(-1, 1)

          hood = hoods[bmu_idx]
          hoods_accum = hoods_accum + hood  
          updates[i] = ex * hood

        frame_callback and frame_callback(t, ex, hood, som)

        som = np.divide(np.sum(updates, axis=0), hoods_accum + 1e-8)
        t = t + 1

      frame_callback and frame_callback(t, ex, hood, som)
      return som
    return (train_som_batch,)


@app.cell
def _(np, train_som_batch):
    bfc = HistoryCallback(240,135,3, lambda x: True)
    color_som_batch =\
      train_som_batch(np.random.random(size=(1000, 3)), 
                       240, 135, 
                       120, 70, 
                       50, min_sigma_frac=.2, seed=None, frame_callback=bfc)
    return (color_som_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Mouse over the above cell.  How long did it take to execute?

    ✅ You can inspect the map in the next cell.  The batch algorithm will almost certainly have a different result than the online algorithm.  Does it matter?  Are there qualitative differences between the mappings?  How would you quantify these differences, and for which applications would they be relevant?
    """
    )
    return


@app.cell
def _(color_som_batch, px):
    px.imshow(color_som_batch.reshape(240,135,3).swapaxes(0,1)).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Once you're ready to move on, click through to [the final notebook](./?file=som-jax.py).""")
    return


if __name__ == "__main__":
    app.run()
