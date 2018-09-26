# -*- coding: utf-8 -*-
from collections import OrderedDict
import itertools
from math import sqrt
import operator
import sys

import numpy as np

from .learner2D import Learner2D, default_loss, choose_point_in_triangle, areas


def unpack_point(point):
    return tuple(point[0]), point[1]


class AverageLearner2D(Learner2D):
    def __init__(self, function, bounds, weight=1, loss_per_triangle=None):
        """Same as 'Learner2D', only the differences are in the doc-string.

        Parameters
        ----------
        function : callable
            The function to learn. Must take a tuple of a tuple of two real
            parameters and a seed and return a real number.
            So ((x, y), seed) → float, e.g.:
            >>> def f(xy_seed):
            ...     (x, y), seed = xy_seed
            ...     return x * y + random(seed)
        weight : float, int, default 1
            When `weight > 1` adding more points to existing points will be
            prioritized (making the standard error of a point more imporant,)
            otherwise adding new triangles will be prioritized (making the 
            loss of a triangle more important.)

        Attributes
        ----------
        min_values_per_point : int, default 3
            Minimum amount of values per point. This means that the
            standard error of a point is infinity until there are
            'min_values_per_point' for a point.

        Methods
        -------
        mean_values_per_point : callable
            Returns the average numbers of values per (x, y) value.

        Notes
        -----
        The total loss of the learner is still only determined by the
        max loss of the triangles.
        """

        super().__init__(function, bounds, loss_per_triangle)
        self._data = dict()  # {point: {seed: value}} mapping
        self.pending_points = dict()  # {point: {seed}}

        # Adding a seed of 0 to the _stack to
        # make {((x, y), seed): loss_improvements, ...}.
        self._stack = {(p, 0): l for p, l in self._stack.items()}
        self.weight = weight
        self.min_values_per_point = 3

    def standard_error(self, lst):
        n = len(lst)
        if n < self.min_values_per_point:
            return sys.float_info.max
        sum_f_sq = sum(x**2 for x in lst)
        mean = sum(x for x in lst) / n
        numerator = sum_f_sq - n * mean**2
        if numerator < 0:
            # This means that the numerator is ~ -1e-15
            return 0
        std = sqrt(numerator / (n - 1))
        return std / sqrt(n)

    def mean_values_per_point(self):
        return np.mean([len(x.values()) for x in self._data.values()])

    @property
    def bounds_are_done(self):
        return all(p in self.data for p in self._bounds_points)

    @property
    def data(self):
        return {k: sum(v.values()) / len(v) for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: self.standard_error(v.values()) for k, v in self._data.items()}

    def _add_to_pending(self, point):
        xy, seed = unpack_point(point)
        if xy not in self.pending_points:
            self.pending_points[xy] = set()
        self.pending_points[xy].add(seed)

    def _remove_from_to_pending(self, point):
        xy, seed = unpack_point(point)
        if xy in self.pending_points:
            self.pending_points[xy].discard(seed)
            if not self.pending_points[xy]:
                # pending_points[xy] is now empty so delete the set()
                del self.pending_points[xy]

    def _add_to_data(self, point, value):
        xy, seed = unpack_point(point)
        if xy not in self._data:
            self._data[xy] = {}
        self._data[xy][seed] = value

    def get_seed(self, point):
        _data = self._data.get(point, {})
        pending_seeds = self.pending_points.get(point, set())
        seed = len(_data) + len(pending_seeds)
        if seed in _data or seed in pending_seeds:
            # means that the seed already exists, for example
            # when '_data[point].keys() | pending_points[point] == {0, 2}'.
            return (set(range(seed)) - pending_seeds - _data.keys()).pop()
        return seed

    def loss_per_existing_point(self):
        if self.data:
            _, values = self._data_in_bounds()
            z_scale = values.ptp()
            z_scale = z_scale if z_scale > 0 else 1
        else:
            z_scale = 1

        points = []
        loss_improvements = []
        for p, sem in self.data_sem.items():
            if sem > 0:
                points.append((p, self.get_seed(p)))
                N = len(self._data[p])
                sem_improvement = (1 - sqrt(N - 1) / sqrt(N)) * sem
                loss_improvements.append(self.weight * sem_improvement / z_scale)
        return points, loss_improvements

    def ask(self, n, tell_pending=True):
        points, loss_improvements = super().ask(n, tell_pending=False)

        p, l = self.loss_per_existing_point()
        points += p
        loss_improvements += l

        loss_improvements, points = zip(*sorted(
            zip(loss_improvements, points), reverse=True))

        points = list(points)[:n]
        loss_improvements = list(loss_improvements)[:n]

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def inside_bounds(self, xy_seed):
        xy, seed = unpack_point(xy_seed)
        return super().inside_bounds(xy)

    def modify_point(self, point):
        """Adding a point with seed = 0.
        This used in '_fill_stack'."""
        return (tuple(point), 0)

    def remove_unfinished(self):
        self.pending_points = {}
        for p in self._bounds_points:
            if p not in self.data:
                self._stack[(p, 0)] = np.inf

    def plot_std_or_n(self, which='std'):
        """Plot the number of points or standard deviation.

        Parameters
        ----------
        which : str
            'n' or 'std'.

        Returns
        -------
        plot : hv.Image
            Plot of the 'number of points' or 'std' per point.
        """
        assert which in ('n', 'std')
        tmp_learner = Learner2D(lambda _: _, bounds=self.bounds)
        f = lambda x: len(x) if which == 'n' else np.std(list(x.values()))
        tmp_learner._data = {k: f(v) for k, v in self._data.items()}
        return tmp_learner.plot()
