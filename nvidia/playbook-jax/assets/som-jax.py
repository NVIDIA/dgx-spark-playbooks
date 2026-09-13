# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax==0.7.2",
#     "marimo",
#     "numpy==2.2.6",
#     "plotly==6.3.0",
#     "tqdm==4.67.1",
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
def _(mo):
    mo.md(
        r"""
    # Self-organizing maps in JAX

    In this notebook, we'll develop implementations of online and batch self-organizing map training in JAX, refining each as we go to get better performance.  We'll start with the easiest option:  simply using JAX as a drop-in replacement for numpy.

    ## Accelerating NumPy functions with JAX
    """
    )
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np
    return jax, jnp, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Recall that we're initializing our map with random vectors.  The result of this function is a matrix with a row for every element in a self-organizing map; each row contains uniformly-sampled random numbers between 0 and 1.

    Because JAX uses a purely functional approach to random number generation, we'll need to rewrite this code from the numpy implementation -- instead of using a stateful generator like numpy's `Generator` or `RandomState`, we'll create a `PRNGKey` object and pass that to `jax.random.uniform`.  (For this example, we're not doing anything with the key — for a real application, we'd want to _split_ it so we could get the next number in the seeded sequence.)
    """
    )
    return


@app.cell
def _(jax, jnp):
    def init_som(xdim, ydim, fdim, seed):
        key = jax.random.PRNGKey(seed)
        return jnp.array(jax.random.uniform(key, shape=(xdim * ydim * fdim,)).reshape(xdim * ydim, fdim))
    return (init_som,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see that JAX is not returning a numpy array:""")
    return


@app.cell
def _(init_som):
    x_size = 192
    y_size = 108
    feature_dims = 3

    random_map = init_som(x_size, y_size, feature_dims, 42)
    type(random_map)
    return feature_dims, random_map, x_size, y_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...and we should be able to see that this array is stored in GPU memory (if we're actually running on a GPU).""")
    return


@app.cell
def _(random_map):
    random_map.device
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As before, you can visualize the result if you want — JAX will transfer arrays directly to device memory when needed by plotting libraries.""")
    return


@app.cell
def _(feature_dims, random_map, x_size, y_size):
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default='notebook'

    px.imshow(random_map.reshape(x_size, y_size, feature_dims).swapaxes(0,1))
    return (px,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our neighborhood function is very similar to the numpy implementation; the only difference is that we need to change `np` to `jnp`.""")
    return


@app.cell
def _(jnp):
    def neighborhood(range_x, range_y, center_x, center_y, x_sigma, y_sigma):
      x_distance = jnp.abs(center_x - range_x)
      x_neighborhood = jnp.exp(- jnp.square(x_distance) / jnp.square(x_sigma))

      y_distance = jnp.abs(center_y - range_y)
      y_neighborhood = jnp.exp(- jnp.square(y_distance) / jnp.square(y_sigma))

      return jnp.outer(x_neighborhood, y_neighborhood)
    return (neighborhood,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plotting results is a good way to make sure that they look the way we expect them to.""")
    return


@app.cell
def _(neighborhood, np, px, x_size, y_size):
    center_x = 12
    center_y = 48
    sigma_x = 96
    sigma_y = 54

    px.imshow(neighborhood(np.arange(x_size), np.arange(y_size), center_x, center_y, sigma_x, sigma_y).T)
    return center_x, center_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We're now ready to see the basic online (i.e., one sample at a time) training algorithm.  Most of it is unchanged from the numpy implementation, with a few key differences:

    1. The first differences are related to how we shuffle the example array. Because we aren't using a stateful random number generator, we'll need to split the random state key into two parts (one representing the key for the very next generation and one representing the key for the rest of the stream).  We'll declare a little helper function that splits the key, shuffles the array, and returns both the key and the shuffled array.
    2. The second difference relates to how JAX handles arrays.  In JAX, arrays offer an _immutable_ interface:  instead of changing an array directly, JAX's API lets you make a copy of the array with a change.  (In practice, this does not always mean the array is actually copied!)  This impacts our code because the numpy version used some functions with output parameters, which indicate where to write the return value (rather than merely returning a new array).  So, instead of `np.add(a, b, a)`, we'd do `a = jnp.add(a, b)`.
    """
    )
    return


@app.cell
def _(init_som, jax, jnp, neighborhood):
    def shuffle(key, examples):
        key, nextkey = jax.random.split(key)
        examples = jax.random.permutation(nextkey, examples)
        return (key, examples)

    def train_som_online(examples, xdim, ydim, x_sigma, y_sigma, max_iter, seed=42, frame_callback=None):
        t = -1
        exs = examples.copy()
        fdim = exs.shape[-1]
        x_sigmas = jnp.linspace(x_sigma, max(5, x_sigma * 0.15), max_iter)
        y_sigmas = jnp.linspace(y_sigma, max(5, y_sigma * 0.15), max_iter)
        alphas = jnp.geomspace(0.35, 0.01, max_iter)
        range_x, range_y = (jnp.arange(xdim), jnp.arange(ydim))
        hood = None
        som = init_som(xdim, ydim, fdim, seed)
        key = jax.random.PRNGKey(seed)
        while t < max_iter:
            key, exs = shuffle(key, exs)
            for ex in exs:
                t = t + 1
                if t == max_iter:
                    break
                bmu_idx = jnp.argmin(jnp.linalg.norm(ex - som, axis=1))
                bmu = som[bmu_idx]
                center_x = bmu_idx // ydim
                center_y = bmu_idx % ydim
                hood = neighborhood(range_x, range_y, center_x, center_y, x_sigmas[t], y_sigmas[t]).reshape(-1, 1)
                update = jnp.multiply((ex - som) * alphas[t], hood)
                frame_callback and frame_callback(t - 1, ex, hood, som)
                som = jnp.add(som, update)
                som = jnp.clip(som, 0, 1)
        frame_callback and frame_callback(t, ex, hood, som)
        return som
    return shuffle, train_som_online


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As [in our NumPy version](./?file=numpy-som.py), we'll use a history callback class to track our progress.""")
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
            self.frames[epoch] = (ex, hood, som)
        if meta is not None:
            self.meta[epoch] = meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we'll train a small map on random color data, storing one history snapshot for every 20 examples.""")
    return


@app.cell
def _(jax, train_som_online):
    fc = HistoryCallback(240, 135, 3, lambda x: x % 20 == 0)
    examples = jax.random.uniform(jax.random.PRNGKey(42), shape=(1000, 3))
    color_som = train_som_online(examples, 240, 135, 120, 70, 50000, 42, fc)
    return color_som, examples


@app.cell
def _(color_som, px):
    px.imshow(color_som.reshape(240,135,3).swapaxes(0,1)).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Mouse over the above cell.  How long did it take to execute?

    Depending on your computer, this may have actually been slower than the numpy version!  Let's try using JAX's _just-in-time_ compilation to improve our performance.  We'll make just-in-time compiled versions of our `neighborhood` and `shuffle` functions (as well as of the inner part of the training loop).  We'll also add a progress bar.
    """
    )
    return


@app.cell
def _(center_x, center_y, init_som, jax, jnp, neighborhood, shuffle):
    import tqdm
    jit_neighborhood = jax.jit(neighborhood)
    jit_shuffle = jax.jit(shuffle)

    @jax.jit
    def som_step(ex, som, xdim, ydim, range_x, range_y, center_x, center_y, x_sigma, y_sigma, alpha):
        bmu_idx = jnp.argmin(jnp.linalg.norm(ex - som, axis=1))
        bmu = som[bmu_idx]
        center_x, center_y = jnp.divmod(bmu_idx, ydim)
        hood = jit_neighborhood(range_x, range_y, center_x, center_y, x_sigma, y_sigma).reshape(-1, 1)
        update = jnp.multiply((ex - som) * alpha, hood)
        return jnp.clip(jnp.add(som, update), 0, 1)

    def train_som_online2(examples, xdim, ydim, x_sigma, y_sigma, max_iter, seed=42, frame_callback=None):
        t = 0
        exs = examples.copy()
        fdim = exs.shape[-1]
        x_sigmas = jnp.linspace(x_sigma, max(5, x_sigma * 0.2), max_iter)
        y_sigmas = jnp.linspace(y_sigma, max(5, y_sigma * 0.2), max_iter)
        alphas = jnp.geomspace(0.35, 0.01, max_iter)
        range_x, range_y = (jnp.arange(xdim), jnp.arange(ydim))
        hood = None
        som = init_som(xdim, ydim, fdim, seed)
        key = jax.random.PRNGKey(seed)
        with tqdm.tqdm(total=max_iter) as progress:
            while t < max_iter:
                key, exs = jit_shuffle(key, exs)
                for ex in exs:
                    t = t + 1
                    progress.update(1)
                    if t == max_iter:
                        break
                    som = som_step(ex, som, xdim, ydim, range_x, range_y, center_x, center_y, x_sigmas[t], y_sigmas[t], alphas[t])
                    frame_callback and frame_callback(t, ex, hood, som)
        return som
    return jit_neighborhood, tqdm, train_som_online2


@app.cell
def _(jax, train_som_online2):
    _fc = HistoryCallback(240, 135, 3, lambda x: x % 20 == 0)
    _examples = jax.random.uniform(jax.random.PRNGKey(42), shape=(1000, 3))
    color_som_1 = train_som_online2(_examples, 240, 135, 120, 70, 50000, 42, _fc)
    return (color_som_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Mouse over the above cell.  How long did it take to execute?

    Let's check our final map to make sure it looks somewhat reasonable.
    """
    )
    return


@app.cell
def _(color_som_1, px):
    px.imshow(color_som_1.reshape(240, 135, 3).swapaxes(0, 1)).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ One challenging aspect of the online algorithm is its sensitivity to hyperparameter settings:

    * Try running the code again with some different values for `x_sigma` and `y_sigma` and see how your results change!  (Consider a minimum size for this value based on the size of the map and the number of training examples.)
    * The `alphas` variable (which we didn't expose as a parameter) indicates how much of an effect each example has on the map.  We've set it to `jnp.geomspace(0.35, 0.01, max_iter)`; try some different values and see if you get better or worse results!

    Let's now consider the batch variant of the algorithm.  It can be much faster, can be implemented in parallel (or even on a cluster) and is less sensitive to hyperparameter settings.  In order to exploit additional parallelism, we're going to use `jax.vmap` to calculate weight updates for each training example in parallel.  This should result in a dramatic performance improvement.
    """
    )
    return


@app.cell
def _(init_som, jax, jit_neighborhood, jnp, tqdm):
    from functools import partial

    @partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None), out_axes=0)
    def batch_step(ex, som, range_x, range_y, xdim, ydim, x_sigma, y_sigma):
        bmu_idx = jnp.argmin(((ex - som) ** 2).sum(axis=1))
        bmu = som[bmu_idx]
        center_x, center_y = jnp.divmod(bmu_idx, ydim)
        hood = jit_neighborhood(range_x, range_y, center_x, center_y, x_sigma, y_sigma).reshape(-1, 1)
        return (ex * hood, hood)

    def train_som_batch(examples, xdim, ydim, x_sigma, y_sigma, epochs, min_sigma_frac=0.1, seed=None, frame_callback=None):
        t = 0
        exs = examples.copy()
        fdim = exs.shape[-1]
        x_sigmas = jnp.linspace(x_sigma, max(2, xdim * min_sigma_frac), epochs)
        y_sigmas = jnp.linspace(y_sigma, max(2, ydim * min_sigma_frac), epochs)
        range_x, range_y = (jnp.arange(xdim), jnp.arange(ydim))
        hood = None
        som = init_som(xdim, ydim, fdim, seed)
        for t in tqdm.trange(epochs):
            updates, hoods = batch_step(examples, som, range_x, range_y, xdim, ydim, x_sigmas[t], y_sigmas[t])
            frame_callback and frame_callback(t, None, hood, som)
            som = jnp.divide(jnp.sum(updates, axis=0), jnp.sum(hoods, axis=0).reshape(-1, 1) + 1e-10)
        frame_callback and frame_callback(t, None, hood, som)
        return som
    return (train_som_batch,)


@app.cell
def _(examples, train_som_batch):
    bfc = HistoryCallback(240, 135, 3, lambda x: True)
    color_som_batch = train_som_batch(examples, 240, 135, 120, 70, 50, min_sigma_frac=0.25, seed=42, frame_callback=bfc)
    return (color_som_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Mouse over the above cell.  How long did it take to execute?  How does this compare to our other implementations?

    We have only optimized the batch step here (i.e., we're calculating the best matching unit and map updates for many examples in parallel and then summing these all at once).  There are more opportunities to optimize this code with JAX, but we haven't exploited them in order to make it possible to use code that has side effects within `train_som_batch` -- in particular, we're 

    1. using the `HistoryCallback` buffer so we could debug our implementation if necessary (or render a movie of training), and
    2. using `tqdm` for a nice progress bar.

    ✅ Try removing `tqdm` and `HistoryCallback` and then JIT-compiling `train_som_batch`.  Does the performance improve?  How much?

    ✅ Using JAX looping constructs instead of Python looping (e.g., `for t in tqdm.trange(epochs):`) may enable further optimizations and performance improvements.  Try rewriting the `train_som_batch` to use with JAX's `lax.fori_loop` (use `help` or see the JAX documentation for details). How does the performance change?
    """
    )
    return


@app.cell
def _(color_som_batch, px):
    px.imshow(color_som_batch.reshape(240,135,3).swapaxes(0,1)).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ What would you need to do to rewrite the SOM training to _not_ use `jax.vmap`?  (You don't have to actually implement this unless you're interested in a puzzle!)


    ✅ Modify `batch_step` to use an alternate distance metric.  This will involve modifying the following line of code:   

    ```bmu_idx = jnp.argmin(((ex - som) ** 2).sum(axis=1))```  

    so that you're taking the `argmin` (or `argmax`, if you're looking for similarity!) of a different function over each entry in the map and the current example.  If you don't have a favorite distance or similarity measure, a common example is cosine similarity, which you can calculate for two vectors by dividing their dot product by the product of their magnitudes, like this:
    """
    )
    return


@app.cell
def _(jnp, np):
    example_som = np.random.random(size=(16, 3))
    example_vec = np.random.random(size=(3,))

    jnp.divide(jnp.dot(example_som, example_vec), (jnp.linalg.norm(example_som, axis=1) * jnp.linalg.norm(example_vec)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""✅ If you implemented cosine similarity, what change did you notice to the performance of batch training?  What changes could you make to `train_som_batch` to improve performance?""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
