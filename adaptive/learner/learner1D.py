# -*- coding: utf-8 -*-
from copy import deepcopy
import heapq
import itertools
import math

import numpy as np
import sortedcontainers

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner


def uniform_loss(interval, scale, function_values):
    """Loss function that samples the domain uniformly.

    Works with `~adaptive.Learner1D` only.

    Examples
    --------
    >>> def f(x):
    ...     return x**2
    >>>
    >>> learner = adaptive.Learner1D(f,
    ...                              bounds=(-1, 1),
    ...                              loss_per_interval=uniform_sampling_1d)
    >>>
    """
    x_left, x_right = interval
    x_scale, _ = scale
    dx = (x_right - x_left) / x_scale
    return dx


def default_loss(interval, scale, function_values):
    """Calculate loss on a single interval

    Currently returns the rescaled length of the interval. If one of the
    y-values is missing, returns 0 (so the intervals with missing data are
    never touched. This behavior should be improved later.
    """
    x_left, x_right = interval
    y_right, y_left = function_values[x_right], function_values[x_left]
    x_scale, y_scale = scale
    dx = (x_right - x_left) / x_scale
    if y_scale == 0:
        loss = dx
    else:
        dy = (y_right - y_left) / y_scale
        try:
            _ = len(dy)
            loss = np.hypot(dx, dy).max()
        except TypeError:
            loss = math.hypot(dx, dy)
    return loss


class Learner1D(BaseLearner):
    """Learns and predicts a function 'f:ℝ → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single real parameter and
        return a real number.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    loss_per_interval: callable, optional
        A function that returns the loss for a single interval of the domain.
        If not provided, then a default is used, which uses the scaled distance
        in the x-y plane as the loss. See the notes for more details.

    Notes
    -----
    'loss_per_interval' takes 3 parameters: interval, scale, and function_values,
    and returns a scalar; the loss over the interval.

    interval : (float, float)
        The bounds of the interval.
    scale : (float, float)
        The x and y scale over all the intervals, useful for rescaling the
        interval loss.
    function_values : dict(float -> float)
        A map containing evaluated function values. It is guaranteed
        to have values for both of the points in 'interval'.
    """

    def __init__(self, function, bounds, loss_per_interval=None):
        self.function = function
        self.loss_per_interval = loss_per_interval or default_loss

        # A dict storing the loss function for each interval x_n.
        self.losses = {}
        self.losses_combined = {}

        self.data = sortedcontainers.SortedDict()
        self.pending_points = set()

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors = sortedcontainers.SortedDict()
        self.neighbors_combined = sortedcontainers.SortedDict()

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [list(bounds), [np.inf, -np.inf]]

        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [bounds[1] - bounds[0], 0]
        self._oldscale = deepcopy(self._scale)

        # The precision in 'x' below which we set losses to 0.
        self._dx_eps = 2 * max(np.abs(bounds)) * np.finfo(float).eps

        self.bounds = list(bounds)

        self._vdim = None

    @property
    def vdim(self):
        return 1 if self._vdim is None else self._vdim

    @property
    def npoints(self):
        return len(self.data)

    def loss(self, real=True):
        losses = self.losses if real else self.losses_combined
        return max(losses.values()) if len(losses) > 0 else float('inf')

    def update_interpolated_loss_in_interval(self, x_left, x_right):
        if x_left is not None and x_right is not None:
            dx = x_right - x_left
            if dx < self._dx_eps:
                loss = 0
            else:
                loss = self.loss_per_interval((x_left, x_right),
                                              self._scale, self.data)
            self.losses[x_left, x_right] = loss

            start = self.neighbors_combined.bisect_left(x_left)
            end = self.neighbors_combined.bisect_left(x_right)
            for i in range(start, end):
                keys = self.neighbors_combined.keys()
                a, b = (keys[i], keys[i + 1])
                self.losses_combined[a, b] = (b - a) * loss / dx
            if start == end:
                self.losses_combined[x_left, x_right] = loss

    def update_losses(self, x, real=True):
        # when we add a new point x, we should update the losses
        x_left, x_right = self.find_neighbors(x, self.neighbors)
        a, b = self.find_neighbors(x, self.neighbors_combined)

        # if (a, b) exists, then (a, b) is splitted into (a, x) and (x, b) 
        self.losses_combined.pop((a, b), None)  # so we get rid of (a, b)
        
        if real:
            # We need to update all interpolated losses in the interval 
            # (x_left, x) and (x, x_right). Since the addition of the point x 
            # could change their loss
            self.update_interpolated_loss_in_interval(x_left, x)
            self.update_interpolated_loss_in_interval(x, x_right)

            # since x goes in between (x_left, x_right), 
            # we get rid of the interval 
            self.losses.pop((x_left, x_right), None)
            self.losses_combined.pop((x_left, x_right), None)
        else:
            if x_left is not None and x_right is not None:
                # x happens to be in between two real points, 
                # so we can interpolate the losses
                dx = x_right - x_left
                loss = self.losses[x_left, x_right]
                self.losses_combined[a, x] = (x - a) * loss / dx
                self.losses_combined[x, b] = (b - x) * loss / dx

        # (no real point left of x) or (no real point right of a)
        left_loss_is_unknown = (x_left is None) or
                               (not real and x_right is None)
        if (a is not None) and left_loss_is_unknown:
            self.losses_combined[a, x] = float('inf')

        # (no real point right of x) or (no real point left of b)
        right_loss_is_unknown = (x_right is None) or
                                (not real and x_left is None)
        if (b is not None) and right_loss_is_unknown:
            self.losses_combined[x, b] = float('inf')

        

    def find_neighbors(self, x, neighbors):
        if x in neighbors:
            return neighbors[x]
        pos = neighbors.bisect_left(x)
        keys = neighbors.keys()
        x_left = keys[pos - 1] if pos != 0 else None
        x_right = keys[pos] if pos != len(neighbors) else None
        return x_left, x_right

    def update_neighbors(self, x, neighbors):
        if x not in neighbors:  # The point is new
            x_left, x_right = self.find_neighbors(x, neighbors)
            neighbors[x] = [x_left, x_right]
            neighbors.get(x_left, [None, None])[1] = x
            neighbors.get(x_right, [None, None])[0] = x

    def update_scale(self, x, y):
        """Update the scale with which the x and y-values are scaled.

        For a learner where the function returns a single scalar the scale
        is determined by the peak-to-peak value of the x and y-values.

        When the function returns a vector the learners y-scale is set by
        the level with the the largest peak-to-peak value.
         """
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
        if y is not None:
            if self.vdim > 1:
                try:
                    y_min = np.min([self._bbox[1][0], y], axis=0)
                    y_max = np.max([self._bbox[1][1], y], axis=0)
                except ValueError:
                    # Happens when `_bbox[1]` is a float and `y` a vector.
                    y_min = y_max = y
                self._bbox[1] = [y_min, y_max]
                self._scale[1] = np.max(y_max - y_min)
            else:
                self._bbox[1][0] = min(self._bbox[1][0], y)
                self._bbox[1][1] = max(self._bbox[1][1], y)
                self._scale[1] = self._bbox[1][1] - self._bbox[1][0]

    def tell(self, x, y):
        if x in self.data:
            # The point is already evaluated before
            return

        real = y is not None
        if real:
            # either it is a float/int, if not, try casting to a np.array
            if not isinstance(y, (float, int)):
                y = np.asarray(y, dtype=float)

            # Add point to the real data dict
            self.data[x] = y
            # remove from set of pending points
            self.pending_points.discard(x)

            if self._vdim is None:
                try:
                    self._vdim = len(np.squeeze(y))
                except TypeError:
                    self._vdim = 1
        else:
            # The keys of pending_points are the unknown points
            self.pending_points.add(x)

        # Update the neighbors
        self.update_neighbors(x, self.neighbors_combined)
        if real:
            self.update_neighbors(x, self.neighbors)

        # Update the scale
        self.update_scale(x, y)

        # Update the losses
        self.update_losses(x, real)

        # If the scale has increased enough, recompute all losses.
        if self._scale[1] > self._oldscale[1] * 2:

            for interval in self.losses:
                self.update_interpolated_loss_in_interval(*interval)

            self._oldscale = deepcopy(self._scale)

    def ask(self, n, add_data=True):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        if n == 0:
            return [], []

        # If the bounds have not been chosen yet, we choose them first.
        missing_bounds = [b for b in self.bounds if b not in self.data
                          and b not in self.pending_points]

        if missing_bounds:
            loss_improvements = [np.inf] * n
            # XXX: should check if points are present in self.data or self.pending_points
            points = np.linspace(*self.bounds, n + 2 - len(missing_bounds)).tolist()
            if len(missing_bounds) == 1:
                points = points[1:] if missing_bounds[0] == self.bounds[1] else points[:-1]
        else:
            def xs(x_left, x_right, n):
                if n == 1:
                    # This is just an optimization
                    return []
                else:
                    step = (x_right - x_left) / n
                    return [x_left + step * i for i in range(1, n)]

            # Calculate how many points belong to each interval.
            x_scale = self._scale[0]
            quals = [((-loss if not math.isinf(loss) else -(x[1] - x[0]) / x_scale, x, 1))
                     for x, loss in self.losses_combined.items()]
            heapq.heapify(quals)

            for point_number in range(n):
                quality, x, n = quals[0]
                if abs(x[1] - x[0]) / (n + 1) <= self._dx_eps:
                    # The interval is too small and should not be subdivided
                    quality = np.inf
                heapq.heapreplace(quals, (quality * n / (n + 1), x, n + 1))

            points = list(itertools.chain.from_iterable(
                xs(*x, n) for quality, x, n in quals))

            loss_improvements = list(itertools.chain.from_iterable(
                                     itertools.repeat(-quality, n - 1)
                                     for quality, x, n in quals))

        if add_data:
            self.tell_many(points, itertools.repeat(None))

        return points, loss_improvements

    def plot(self):
        hv = ensure_holoviews()
        if not self.data:
            p = hv.Scatter([]) * hv.Path([])
        elif not self.vdim > 1:
            p = hv.Scatter(self.data) * hv.Path([])
        else:
            xs = list(self.data.keys())
            ys = list(self.data.values())
            p = hv.Path((xs, ys)) * hv.Scatter([])

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))

    def remove_unfinished(self):
        self.pending_points = set()
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)
