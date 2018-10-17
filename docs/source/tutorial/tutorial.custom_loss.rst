Custom adaptive logic for 1D and 2D
-----------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`custom-loss-function`

.. execute::
    :hide-code:
    :new-notebook: custom-loss-function

    import adaptive
    adaptive.notebook_extension()

    # Import modules that are used in multiple cells
    import numpy as np
    from functools import partial


`~adaptive.Learner1D` and `~adaptive.Learner2D` both work on the principle of
subdividing their domain into subdomains, and assigning a property to
each subdomain, which we call the *loss*. The algorithm for choosing the
best place to evaluate our function is then simply *take the subdomain
with the largest loss and add a point in the center, creating new
subdomains around this point*.

The *loss function* that defines the loss per subdomain is the canonical
place to define what regions of the domain are “interesting”. The
default loss function for `~adaptive.Learner1D` and `~adaptive.Learner2D` is sufficient
for a wide range of common cases, but it is by no means a panacea. For
example, the default loss function will tend to get stuck on
divergences.

Both the `~adaptive.Learner1D` and `~adaptive.Learner2D` allow you to specify a *custom
loss function*. Below we illustrate how you would go about writing your
own loss function. The documentation for `~adaptive.Learner1D` and `~adaptive.Learner2D`
specifies the signature that your loss function needs to have in order
for it to work with ``adaptive``.

tl;dr, one can use the following *loss functions* that
**we** already implemented:

+ `adaptive.learner.learner1D.default_loss`
+ `adaptive.learner.learner1D.uniform_loss`
+ `adaptive.learner.learner2D.default_loss`
+ `adaptive.learner.learner2D.uniform_loss`
+ `adaptive.learner.learner2D.minimize_triangle_surface_loss`
+ `adaptive.learner.learner2D.resolution_loss`


Uniform sampling
~~~~~~~~~~~~~~~~

Say we want to properly sample a function that contains divergences. A
simple (but naive) strategy is to *uniformly* sample the domain:

.. execute::

    def uniform_sampling_1d(interval, scale, function_values):
        # Note that we never use 'function_values'; the loss is just the size of the subdomain
        x_left, x_right = interval
        x_scale, _ = scale
        dx = (x_right - x_left) / x_scale
        return dx

    def f_divergent_1d(x):
        return 1 / x**2

    learner = adaptive.Learner1D(f_divergent_1d, (-1, 1), loss_per_interval=uniform_sampling_1d)
    runner = adaptive.BlockingRunner(learner, goal=lambda l: l.loss() < 0.01)
    learner.plot().select(y=(0, 10000))

.. execute::

    %%opts EdgePaths (color='w') Image [logz=True]

    from adaptive.runner import SequentialExecutor

    def uniform_sampling_2d(ip):
        from adaptive.learner.learner2D import areas
        A = areas(ip)
        return np.sqrt(A)

    def f_divergent_2d(xy):
        x, y = xy
        return 1 / (x**2 + y**2)

    learner = adaptive.Learner2D(f_divergent_2d, [(-1, 1), (-1, 1)], loss_per_triangle=uniform_sampling_2d)

    # this takes a while, so use the async Runner so we know *something* is happening
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.02)

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. execute::

    runner.live_info()
    runner.live_plot(update_interval=0.2,
                     plotter=lambda l: l.plot(tri_alpha=0.3).relabel('1 / (x^2 + y^2) in log scale'))

The uniform sampling strategy is a common case to benchmark against, so
the 1D and 2D versions are included in ``adaptive`` as
`adaptive.learner.learner1D.uniform_loss` and
`adaptive.learner.learner2D.uniform_loss`.

Doing better
~~~~~~~~~~~~

Of course, using ``adaptive`` for uniform sampling is a bit of a waste!

Let’s see if we can do a bit better. Below we define a loss per
subdomain that scales with the degree of nonlinearity of the function
(this is very similar to the default loss function for `~adaptive.Learner2D`),
but which is 0 for subdomains smaller than a certain area, and infinite
for subdomains larger than a certain area.

A loss defined in this way means that the adaptive algorithm will first
prioritise subdomains that are too large (infinite loss). After all
subdomains are appropriately small it will prioritise places where the
function is very nonlinear, but will ignore subdomains that are too
small (0 loss).

.. execute::

    %%opts EdgePaths (color='w') Image [logz=True]

    def resolution_loss(ip, min_distance=0, max_distance=1):
        """min_distance and max_distance should be in between 0 and 1
        because the total area is normalized to 1."""

        from adaptive.learner.learner2D import areas, deviations

        A = areas(ip)

        # 'deviations' returns an array of shape '(n, len(ip))', where
        # 'n' is the  is the dimension of the output of the learned function
        # In this case we know that the learned function returns a scalar,
        # so 'deviations' returns an array of shape '(1, len(ip))'.
        # It represents the deviation of the function value from a linear estimate
        # over each triangular subdomain.
        dev = deviations(ip)[0]

        # we add terms of the same dimension: dev == [distance], A == [distance**2]
        loss = np.sqrt(A) * dev + A

        # Setting areas with a small area to zero such that they won't be chosen again
        loss[A < min_distance**2] = 0

        # Setting triangles that have a size larger than max_distance to infinite loss
        loss[A > max_distance**2] = np.inf

        return loss

    loss = partial(resolution_loss, min_distance=0.01)

    learner = adaptive.Learner2D(f_divergent_2d, [(-1, 1), (-1, 1)], loss_per_triangle=loss)
    runner = adaptive.BlockingRunner(learner, goal=lambda l: l.loss() < 0.02)
    learner.plot(tri_alpha=0.3).relabel('1 / (x^2 + y^2) in log scale')

Awesome! We zoom in on the singularity, but not at the expense of
sampling the rest of the domain a reasonable amount.

The above strategy is available as
`adaptive.learner.learner2D.resolution_loss`.
