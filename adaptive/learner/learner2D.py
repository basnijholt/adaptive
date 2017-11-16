# -*- coding: utf-8 -*-
import collections
from copy import deepcopy
import itertools

import holoviews as hv
import numpy as np
from scipy import interpolate, spatial

from .base_learner import BaseLearner
from .utils import restore


# Learner2D and helper functions.

def deviations(ip):
    values = ip.values / (ip.values.ptp() or 1)
    gradients = interpolate.interpnd.estimate_gradients_2d_global(
        ip.tri, values, tol=1e-6)

    p = ip.tri.points[ip.tri.vertices]
    vs = values[ip.tri.vertices]
    gs = gradients[ip.tri.vertices]

    def deviation(p, v, g):
        dev = 0
        for j in range(3):
            vest = v[:, j, None] + ((p[:, :, :] - p[:, j, None, :]) *
                                    g[:, j, None, :]).sum(axis=-1)
            dev += abs(vest - v).max(axis=1)
        return dev

    n_levels = vs.shape[2]
    devs = [deviation(p, vs[:, :, i], gs[:, :, i]) for i in range(n_levels)]
    return devs


def areas(ip):
    p = ip.tri.points[ip.tri.vertices]
    q = p[:, :-1, :] - p[:, -1, None, :]
    areas = abs(q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0]) / 2
    return areas


def _default_loss_per_triangle(ip):
    devs = deviations(ip)
    area_per_triangle = np.sqrt(areas(ip))
    losses = np.sum([dev * area_per_triangle for dev in devs], axis=0)
    return losses


class Learner2D(BaseLearner):
    """Learns and predicts a function 'f: ℝ^2 → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of two real
        parameters and return a real number.
    bounds : list of 2-tuples
        A list ``[(a1, b1), (a2, b2)]`` containing bounds,
        one per dimension.
    loss_per_triangle : callable, optional
        A function that returns the loss for every triangle.
        If not provided, then a default is used, which uses
        the deviation from a linear estimate, as well as
        triangle area, to determine the loss. See the notes
        for more details.


    Attributes
    ----------
    points_combined
        Sample points so far including the unknown interpolated ones.
    values_combined
        Sampled values so far including the unknown interpolated ones.
    points
        Sample points so far with real results.
    values
        Sampled values so far with real results.

    Notes
    -----
    Adapted from an initial implementation by Pauli Virtanen.

    The sample points are chosen by estimating the point where the
    linear and cubic interpolants based on the existing points have
    maximal disagreement. This point is then taken as the next point
    to be sampled.

    In practice, this sampling protocol results to sparser sampling of
    smooth regions, and denser sampling of regions where the function
    changes rapidly, which is useful if the function is expensive to
    compute.

    This sampling procedure is not extremely fast, so to benefit from
    it, your function needs to be slow enough to compute.

    'loss_per_triangle' takes a single parameter, 'ip', which is a
    `scipy.interpolate.LinearNDInterpolator`. You can use the
    *undocumented* attributes 'tri' and 'values' of 'ip' to get a
    `scipy.spatial.Delaunay` and a vector of function values.
    These can be used to compute the loss. The functions
    `adaptive.learner.learner2D.areas` and
    `adaptive.learner.learner2D.deviations` to calculate the
    areas and deviations from a linear interpolation
    over each triangle.
    """

    def __init__(self, function, bounds, loss_per_triangle=None):
        self.ndim = len(bounds)
        self.loss_per_triangle = loss_per_triangle or _default_loss_per_triangle
        self._vdim = None
        if self.ndim != 2:
            raise ValueError("Only 2-D sampling supported.")
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.data = collections.OrderedDict()
        self.data_combined = collections.OrderedDict()
        self._stack = collections.OrderedDict()
        self._interp = set()

        xy_mean = np.mean(self.bounds, axis=1)
        xy_scale = np.ptp(self.bounds, axis=1)

        def scale(points):
            return (points - xy_mean) / xy_scale

        def unscale(points):
            return points * xy_scale + xy_mean

        self.scale = scale
        self.unscale = unscale

        self._bounds_points = list(itertools.product(*bounds))

        self._tri = None
        self.tri_combined = spatial.Delaunay(self.scale(self._bounds_points),
                                             incremental=True,
                                             qhull_options='Q11 QJ')

        for point in self._bounds_points:
            self.data_combined[point] = None
            self._stack[point] = np.inf
            self._interp.add(point)

        self.function = function

    @property
    def vdim(self):
        return 1 if self._vdim is None else self._vdim

    @property
    def points_combined(self):
        return np.array(list(self.data_combined.keys()))

    @property
    def values_combined(self):
        return np.array(list(self.data_combined.values()))

    @property
    def points(self):
        return np.array(list(self.data.keys()))

    @property
    def values(self):
        return np.array(list(self.data.values()))

    @property
    def n(self):
        return len(self.data_combined)

    @property
    def n_real(self):
        return len(self.data)

    @property
    def bounds_are_done(self):
        return not any(p in self._interp for p in self._bounds_points)

    @property
    def tri(self):
        if self._tri is None:
            self._tri = spatial.Delaunay(self.scale(self.points),
                                         incremental=True,
                                         qhull_options='Q11 QJ')
        return self._tri

    def ip(self):
        return interpolate.LinearNDInterpolator(self.tri, self.values)

    def ip_combined(self):
        # Interpolate the unfinished points
        if self._interp:
            points_interp = list(self._interp)
            if self.bounds_are_done:
                values_interp = self.ip()(points_interp)
            else:
                # It is important not to return exact zeros because
                # otherwise the algo will try to add the same point
                # to the stack each time.
                values_interp = np.random.rand(
                    len(points_interp), self.vdim) * 1e-15

            for point, value in zip(points_interp, values_interp):
                assert point in self.data_combined
                self.data_combined[point] = value

        return interpolate.LinearNDInterpolator(self.tri_combined,
                                                self.values_combined)

    def add_point(self, point, value):
        point = tuple(point)
        self.data_combined[point] = value

        if value is None:
            if point not in self._bounds_points:
                self.tri_combined.add_points([self.scale(point)])
                self._interp.add(point)
        else:
            if self.bounds_are_done:
                self.tri.add_points([self.scale(point)])
            self.data[point] = value
            self._interp.discard(point)

        self._stack.pop(point, None)

    def _fill_stack(self, stack_till=None):
        if stack_till is None:
            stack_till = 1

        if len(self.data_combined) < self.ndim + 1:
            raise ValueError("too few points...")

        # Interpolate
        ip = self.ip_combined()
        tri = ip.tri

        losses = self.loss_per_triangle(ip)

        def point_exists(p):
            eps = np.finfo(float).eps * self.points_combined.ptp() * 100
            if abs(p - self.points_combined).sum(axis=1).min() < eps:
                return True
            if self._stack:
                _stack_points, _ = self._split_stack()
                if abs(p - np.asarray(_stack_points)).sum(axis=1).min() < eps:
                    return True
            return False

        for j, _ in enumerate(losses):
            # Estimate point of maximum curvature inside the simplex
            jsimplex = np.argmax(losses)
            p = tri.points[tri.vertices[jsimplex]]
            point_new = self.unscale(p.mean(axis=-2))

            # XXX: not sure whether this is necessary it was there
            # originally.
            point_new = np.clip(point_new, *zip(*self.bounds))

            # Check if it is really new
            if point_exists(point_new):
                losses[jsimplex] = 0
                continue

            # Add to stack
            self._stack[tuple(point_new)] = losses[jsimplex]

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = 0

    def _split_stack(self, n=None):
        points, loss_improvements = zip(*reversed(self._stack.items()))
        return points[:n], loss_improvements[:n]

    def _choose_and_add_points(self, n):
        if n <= len(self._stack):
            points, loss_improvements = self._split_stack(n)
            self.add_data(points, itertools.repeat(None))
        else:
            points = []
            loss_improvements = []
            n_left = n
            while n_left > 0:
                # The while loop is needed because `stack_till` could be larger
                # than the number of triangles between the points. Therefore
                # it could fill up till a length smaller than `stack_till`.
                no_bounds_in_stack = not any(p in self._stack
                                             for p in self._bounds_points)
                if no_bounds_in_stack:
                    self._fill_stack(stack_till=n_left)
                new_points, new_loss_improvements = self._split_stack(n_left)
                points += new_points
                loss_improvements += new_loss_improvements
                self.add_data(new_points, itertools.repeat(None))
                n_left -= len(new_points)

        return points, loss_improvements

    def choose_points(self, n, add_data=True):
        if not add_data:
            with restore(self):
                return self._choose_and_add_points(n)
        else:
            return self._choose_and_add_points(n)

    def loss(self, real=True):
        n = self.n_real if real else self.n
        if n <= 4 or not self.bounds_are_done:
            return np.inf
        ip = self.ip() if real else self.ip_combined()
        losses = self.loss_per_triangle(ip)
        return losses.max()

    def remove_unfinished(self):
        self.data_combined = deepcopy(self.data)
        self.tri_combined = spatial.Delaunay(self.scale(self.points),
                                             incremental=True,
                                             qhull_options='Q11 QJ')
        self._interp = set()

    def plot(self, n_x=201, n_y=201, triangles_alpha=0):
        if self.vdim > 1:
            raise NotImplemented('holoviews currently does not support',
                                 '3D surface plots in bokeh.')
        x, y = self.bounds
        lbrt = x[0], y[0], x[1], y[1]
        if self.n_real >= 4:
            x = np.linspace(-0.5, 0.5, n_x)
            y = np.linspace(-0.5, 0.5, n_y)
            ip = self.ip()
            z = ip(x[:, None], y[None, :])
            plot = hv.Image(np.rot90(z), bounds=lbrt)

            if triangles_alpha:
                tri_points = self.unscale(ip.tri.points[ip.tri.vertices])
                contours = hv.Contours([p for p in tri_points])
                contours = contours.opts(style=dict(alpha=triangles_alpha))

        else:
            plot = hv.Image([], bounds=lbrt)
            contours = hv.Contours([])

        return plot * contours if triangles_alpha else plot
