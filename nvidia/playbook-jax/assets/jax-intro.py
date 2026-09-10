# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax==0.7.1",
#     "marimo",
#     "numpy==2.2.6",
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
    # Introducing JAX

    [JAX](https://jax.readthedocs.io) is an implementation of a significant subset of the NumPy API with some features that make it especially suitable for machine learning research and high-performance computing.  As we'll see, these same features also make JAX extremely useful for accelerating functions and prototypes that we've developed in NumPy.  This notebook will provide a quick introduction to just some of the features of JAX that we'll be using in the rest of this workshop -- as well as pointers to some of the potential pitfalls you might run in to with it.  There's a lot more to JAX than we'll be able to cover in this notebook (and the rest of the workshop), so you'll want to read the (great) documentation as you dive in more.

    We'll start by importing JAX and its implementation of the NumPy API.  By convention, we'll import `jax.numpy` as `jnp` and regular `numpy` as `np` — this is because we can use both in our programs and we may want to use both for different things.
    """
    )
    return


@app.cell
def _():
    import jax
    import numpy as np
    import jax.numpy as jnp
    from timeit import timeit
    return jax, jnp, np, timeit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can use `jax.numpy` as a drop-in replacement for `numpy` in many cases (we'll see some caveats later in this notebook).  In many cases, JAX arrays interoperate transparently with NumPy arrays.""")
    return


@app.cell
def _(jnp):
    za = jnp.zeros(7)
    za
    return (za,)


@app.cell
def _(np, za):
    # note that we're doing elementwise addition  
    # between a NumPy array and a JAX array

    zna = np.ones(7)
    za + zna
    return (zna,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There are some differences, though, and an important one is that JAX arrays can be stored in GPU memory.  If you're running this notebook with a GPU-enabled version of JAX, you can see where our array is stored:""")
    return


@app.cell
def _(za):
    za.device
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## NumPy and JAX

    By itself, the JAX implementations of NumPy operations are unlikely to be much faster than those from NumPy or (especially) cuPy, but — as we'll see — JAX offers some special functionality that can make JAX code much faster, especially on a GPU.  Let's start by doing some simple timings of operations, though.

    We'll use the `device_put` method to convert a NumPy array to a JAX array.
    """
    )
    return


@app.cell
def _(jax, np):
    random_shape = (8192, 2048)
    random_values = np.random.random(size=random_shape)
    jrandom_values = jax.device_put(random_values)
    return jrandom_values, random_values


@app.cell
def _(np, random_values):
    np.matmul(random_values, random_values.T)
    return


@app.cell
def _(np, random_values, timeit):
    _result = timeit(lambda: 
        np.matmul(random_values, random_values.T), 
        number=10)

    print(f"NumPy matrix multiplication took {_result/10:.4f} seconds per iteration")
    return


@app.cell
def _(jnp, jrandom_values, timeit):
    _result = timeit(lambda: 
           jnp.matmul(jrandom_values, jrandom_values.T).block_until_ready(), 
           number=10)

    print(f"JAX matrix multiplication took {_result/10:.4f} seconds per iteration")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `block_until_ready` is a special detail that is important when we're getting timings of single lines of JAX code — basically, JAX dispatches our code to the GPU asynchronously and we need to make sure that the operation has completed before we consider it done for the purposes of timing it.

    ✅ Try running the JAX code without `block_until_ready()` to see how much of a difference it makes to time the actual code we want to execute.

    ✅ You probably saw that JAX was faster than NumPy with the matrix shape we provided (in `random_shape`).  Make sure that you've added `.block_until_ready()` back to the JAX code and try some other matrix shapes (both larger and smaller).  Does JAX exhibit more of a speed advantage on some matrix sizes than others?  Is JAX slower than NumPy for some of these?  Why (or why not), do you suppose?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Functional array updates

    One major difference between JAX and NumPy is that JAX arrays are _immutable_.  This means that once you create an array, you can't update it in place.  In NumPy, you'd do this:
    """
    )
    return


@app.cell
def _(zna):
    zna[3] = 5.0
    zna
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...whereas in JAX, you'd need to use some methods to make a copy of the array changing only one value:""")
    return


@app.cell
def _(za):
    za_1 = za.at[3].set(5.0)
    za_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Those of you who have used functional languages will likely be comfortable with this style, but it may be an adjustment.  (It's not necessarily as inefficient as it sounds!  See [here](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation) for more details on how to avoid copies — and what JAX does under the hood.)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Random number generation

    Recall that a pseudorandom number generator is, under the hood, a function that maps from a state to a next state and a value and has a very long period.  The state part is typically hidden from the user as an implementation detail:  you initialize a generator with a seed, it creates an initial state, and then it updates that state every time you draw numbers from the generator.  Implicit state makes parallelism difficult, so JAX takes a different approach -- generator state (a so-called key) is explicitly passed to each method and clients must call a method to split state before drawing from the generator.

    So, in NumPy, drawing some numbers from a Poisson distribution with a &lambda; of 7 would look like this:
    """
    )
    return


@app.cell
def _(np):
    SEED = 0x12345678

    rng = np.random.default_rng(SEED)

    # sample 4,096 values from a Poisson distribution
    npa = rng.poisson(lam=7, size=4096)

    # mean and variance should both be close to 7
    np.mean(npa), np.var(npa)
    return (SEED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...whereas in JAX, it'd look like this:""")
    return


@app.cell
def _(SEED, jax, jnp):
    key = jax.random.PRNGKey(SEED)

    # "split" the key to explicitly manage state -- 
    # both key values will be different from the 
    # key value we generated above

    key, nextkey = jax.random.split(key)

    jnpa = jax.random.poisson(nextkey, lam=7, shape=(4096,))

    # mean and variance should both be close to 7
    jnp.mean(jnpa), jnp.var(jnpa)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""When we split `key`, we generated two new keys.  The advantage of using `nextkey` in the call to `jax.random.poisson` is that we don't have to explicitly assign to `key` later on (e.g., if we were in a loop).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Just-in-time compilation

    If you're running with acclerated hardware, JAX's implementations of NumPy functions require sending code (and sometimes data) to the GPU.  This may not be noticeable if you're doing a lot of work, but it can impact performance if you're invoking many small functions.  JAX provides a method for _just-in-time compilation_ so that the first time you execute a function it produces a specialized version that can execute more efficiently.

    We'll see the `jax.jit` function in action later in this workshop.  There are some things we'll need to keep in mind to use it effectively, and we'll cover those when we get to them!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Parallelizing along axes""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A powerful feature of JAX is the capability to parallelize functions along axes.  So if, for example, you want to calculate the norm of each row in a matrix, you can do each of these in parallel.  We'll start by generating a random matrix and moving it to the GPU again:""")
    return


@app.cell
def _(jax, np):
    random_shape_vmap = (8192, 16384)
    random_values_vmap = np.random.random(size=random_shape_vmap)
    jrandom_values_vmap = jax.device_put(random_values_vmap)
    return jrandom_values_vmap, random_values_vmap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We'll compute the norm of each row using both NumPy and JAX and collect timings for each:""")
    return


@app.cell
def _(np, random_values_vmap, timeit):
    _result = timeit(lambda: 
        np.linalg.norm(random_values_vmap, axis=1), 
        number=10)

    print(f"NumPy row-wise norm took {_result/10:.4f} seconds per iteration")
    return


@app.cell
def _(jnp, jrandom_values_vmap, timeit):
    _result = timeit(lambda: 
        jnp.linalg.norm(jrandom_values_vmap, axis=1).block_until_ready(), 
        number=100)

    print(f"JAX row-wise norm took {_result/100:.7f} seconds per iteration")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also try using `jax.jit` and `jax.vmap` to see how this impacts performance.""")
    return


@app.cell
def _(jax, jnp, jrandom_values_vmap, timeit):
    jit_norm = jax.jit(jax.vmap(jnp.linalg.norm, in_axes=0))

    _result = timeit(lambda: 
        jit_norm(jrandom_values_vmap).block_until_ready(), 
        number=100)

    print(f"JAX vmapped norm took {_result/100:.7f} seconds per iteration")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Calculating the norm of every row is already pretty efficient in JAX, so we don't see much (if any) performance improvement from mapping over axes.  (This will likely be the case for most NumPy functions in JAX that have an `axis` argument.)  However, it's an easy example to understand and we'll see a higher-impact application of `vmap` in the next notebook.

    ## Timings

    In the above cell, we've used the `timeit` module in the standard library to repeatedly execute a small code snippet and get the average execution time.  For longer-executing cells, we can simply use marimo's direct support for recording cell timings to see how long they executed.

    ✅ Mouse over a cell that has executed and look for timing information.  In the version of marimo I'm using now, it will show up in the right margin, but only when you mouse over the cell.  For the cells above, the timing should be roughly the value printed out times the number of iterations in the `timeit` call, so if the cell printed something like:

    ```JAX vmapped norm took 0.0355922 seconds per iteration```

    then you'd expect the cell timing to show something like 3.6 seconds, given that we ran 100 iterations of the code.

    ## Automatic differentiation

    A particularly interesting feature of JAX is its support for _automatic differentiation_.  This means that, given a function $f(x)$, JAX can automatically calculate $f'(x)$, or the _derivative_ of $f$, which is a function describing the rate of change between $f(x)$ and $f(x + \epsilon)$, where $\epsilon$ is a very small number.  (JAX can also calculate the derivative for functions of multiple arguments, but our running example in this notebook will be a single-argument function.)

    If your daily work regularly involves implementing machine learning and optimization algorithms, you probably already have some ideas why this functionality could be useful.  (If it doesn't and you're curious, [here's an explanation](https://en.wikipedia.org/wiki/Gradient_descent) to read on your own time.)  

    In the rest of this notebook, we'll show an example of a problem we can solve with the help of JAX's support for automatic differentiation.  Since not everyone spends their days thinking about optimizing functions, we've chosen a problem that doesn't require any specialized mathematical or machine learning background to understand, but we'll throw in a wrinkle at the end to show everyone why JAX's automatic differentiation is especially cool.

    Let's start with a very simple Python function:
    """
    )
    return


@app.function
def square(x):
    return x * x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can calculate the derivative of `square` numerically, by calculating the slope of `x * x` while making a very small change to `x`.""")
    return


@app.function
def square_num_prime(x, h=1e-5):
    above = x + h
    below = x - h

    rise = (above ** 2) - (below ** 2)
    run = h * 2

    return rise / run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You may remember the [power rule](https://en.wikipedia.org/wiki/Power_rule), which states that the the derivative of $x^a$ is $ax^{a-1}$.  Given this rule, the derivative of $x^2 = 2x^{2-1} = 2x$.  We can use this to check our answer for several values of $x$.""")
    return


@app.cell
def _():
    [square_num_prime(x) for x in [16.0,32.0,64.0,128.0,256.0,512.0,1024.0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You may have noticed that not every result is what we'd expect!  There are several ways in which numerical differentiation may not produce a precise result, but for this example we may be able to improve the results by changing the range around $x$ for which we're measuring the change in $x^2$.

    ✅ Try some different values for the `h` parameter (both larger and smaller) and see if you can get more precise results.
    """
    )
    return


@app.cell
def _():
    H = 1e-9

    [square_num_prime(x, h=H) for x in [16.0,32.0,64.0,128.0,256.0,512.0,1024.0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Numerical differentiation is straightforward and useful, but it can be imprecise when dealing with very large or very small values -- and analogous techniques for functions of multiple variables are far more complicated.  Since machine learning and data processing algorithms often involve functions operating on multiple variables (or vectors of numbers), can involve very large or very small numbers, and may involve repeated calculations that would propagate imprecision, we want a more flexible and less limited technique for differentiation.  Fortunately, JAX provides just this in the form of _automatic differentiation_.

    We'll use the `grad` function to automatically generate the derivative of `square`.
    """
    )
    return


@app.cell
def _(jax):
    # this is trivial but it works!

    square_prime = jax.grad(square)
    square_prime(4.0)
    return (square_prime,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice that `square_prime` returns an array.""")
    return


@app.cell
def _(square_prime):
    square_prime(4.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""... in fact, a _zero-dimensional_ JAX array.  We can access the element with the `item()` function.""")
    return


@app.cell
def _(square_prime):
    [square_prime(x).item() for x in [16.0,32.0,64.0,128.0,256.0,512.0,1024.0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can use `square_prime` to implement [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method#Square_root) for the specific problem of approximating square roots.  Basically the idea is that we'll start with an initial guess as our current guess and then repeatedly:

    1. subtract our goal number (i.e., the number we want the square root for) from our current guess squared,
    2. divide that difference by the derivative of `square` at our current guess, and
    3. update our current guess by subtracting that quotient from it.

    After the third step, we'll compare the square of our guess to our goal number and stop if it's close enough or if we've gone a certain number of iterations.  (You may have used a similar but less-efficient method of iteratively refining guesses for square roots on paper in a primary school arithmetic class!)

    We'll start with a function that updates a guess value given a guess and a goal:
    """
    )
    return


@app.cell
def _(square_prime):
    def guess_sqrt(guess, goal):
        n = ((guess * guess) - goal)
        d = square_prime(guess)
        return guess - (n / d)
    return (guess_sqrt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We'll then build out the whole method, including a tolerance value (i.e., how close does `guess ** 2` need to be to `target` for us to accept it?) and a maximum number of iterations so we don't accidentally get into an infinite loop.""")
    return


@app.cell
def _(guess_sqrt, np):
    def newton_sqrt(initial_guess, target, tolerance=1e-4, max_iter=20):
        guess = initial_guess
        guesses = [initial_guess]

        while np.abs((guess * guess) - target) > tolerance and max_iter > 0:
            guess = guess_sqrt(guess, target)
            guesses.append(guess.item())
            max_iter = max_iter - 1

        return guesses
    return (newton_sqrt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can try this out on a few examples (with some unreasonable initial guesses).""")
    return


@app.cell
def _(newton_sqrt):
    newton_sqrt(3.0, 25.0)
    return


@app.cell
def _(newton_sqrt):
    newton_sqrt(12.0, 65536.0)
    return


@app.cell
def _(newton_sqrt):
    newton_sqrt(256.0, 123456789.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Try some different values for initial guesses and targets and see if you can find some examples that take more or fewer iterations to come to an acceptable solution.

    Automatic differentiation is a cool technology, but — as you may object — differentiating $x^2$ isn't a particularly cool application.  If this were the extent of our requirements, we could simply implement a few rules that inspected Python functions and replaced functions by their derivatives.  If we didn't want to get our hands dirty, we could probably also use a library like `sympy` or hire a first-year undergraduate to perform our calculations for us.

    JAX isn't limited to trivial functions, though.  Let's take a look at a more syntactically (and semantically) complex implementation of the square function.  We'll call it `bogus_square` to emphasize that it is a contrived example that is meant to be difficult for JAX to deal with.
    """
    )
    return


@app.cell
def _(jnp):
    def bogus_square(x):
        result = 0
        for _ in range(int(jnp.floor(x))):
            result = result + x
        return result + x * (x - jnp.floor(x))
    return (bogus_square,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can check our results on some examples:""")
    return


@app.cell
def _(bogus_square):
    square_examples = [1.4142135, 2.0, 4.0, 7.9372539, 8.0, 15.9687194226713, 16.0]

    [bogus_square(x) for x in square_examples]
    return (square_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also differentiate this function, just as we could with the simpler `square`:""")
    return


@app.cell
def _(bogus_square, jax):
    bogus_square_prime = jax.grad(bogus_square)
    return (bogus_square_prime,)


@app.cell
def _(bogus_square_prime, square_examples):
    [bogus_square_prime(x) for x in square_examples]
    return


@app.cell
def _(bogus_square, bogus_square_prime):
    def bogus_guess_sqrt(guess, goal):
        n = (bogus_square(guess) - goal)
        d = bogus_square_prime(guess)
        return guess - (n / d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""✅ Edit the next cell, adapting `newton_sqrt` to use `bogus_square` and `bogus_guess_sqrt`.""")
    return


@app.function
def newton_bogus_sqrt(initial_guess, target, tolerance=1e-4, max_iter=20):
    results = [initial_guess]
    # exercise:  fill in the body of this function
    return results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""✅ Test your function out with a few examples.  You may want to avoid finding square roots of larger numbers (we'll see why in a second).""")
    return


@app.cell
def _():
    newton_bogus_sqrt(8.0, 72.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    It's impressive that JAX can differentiate `bogus_square`, but this doesn't mean that we're free to use pathological implementations in our code.  The derivative of `bogus_square` is much slower to compute than that of `square`, which means that the performance of an iterative process that depends on computing this many times (like machine learning model training or even like approximating square roots) will suffer.

    ✅ Make sure that the two following cells have executed and then mouse over them to see how long they took to execute.
    """
    )
    return


@app.cell
def _(square_prime):
    for _ in range(10):
        square_prime(10000.0)
    return


@app.cell
def _(bogus_square_prime):
    for _ in range(10):
        bogus_square_prime(10000.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ✅ Can you think of a _worse_ way to implement `square`?  Were you able to stump `jax.grad` with it?

    Once you're done here, go on to [the next notebook, where we'll introduce self-organizing maps in numpy](./?file=numpy-som.py).
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
