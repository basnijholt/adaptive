# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
import itertools
from math import sqrt
import operator
import sys

import numpy as np

from .learner2D import Learner2D, default_loss, choose_point_in_triangle, areas


def unpack_point(point):
    return tuple(point[0]), point[1]


class AverageLearner2D(Learner2D):
    def __init__(self, function, bounds, atol=None, rtol=None, loss_per_triangle=None):
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
        self._data = defaultdict(lambda: defaultdict(dict))
        self.pending_points = defaultdict(set)

        # Adding a seed of 0 to the _stack to
        # make {((x, y), seed): loss_improvements, ...}.
        self._stack = {(p, 0): l for p, l in self._stack.items()}
        self.min_values_per_point = 3

        self.atol = atol or np.inf
        self.rtol = rtol or np.inf

    def standard_error(self, lst, normalize=True):
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
        standard_error = std / sqrt(n)
        return max(standard_error / self.atol,
                   standard_error / abs(mean) / self.rtol)

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
        self.pending_points[xy].add(seed)

    def _remove_from_to_pending(self, point):
        xy, seed = unpack_point(point)
        self.pending_points[xy].discard(seed)
        if not self.pending_points[xy]:
            # self.pending_points[xy] is now empty so delete the set()
            del self.pending_points[xy]

    def _add_to_data(self, point, value):
        xy, seed = unpack_point(point)
        self._data[xy][seed] = value

    def get_seed(self, point):
        seed = len(self._data[point]) + len(self.pending_points[point])
        if seed in self._data[point] or seed in self.pending_points[point]:
            # means that the seed already exists, for example
            # when '_data[point].keys() | pending_points[point] == {0, 2}'.
            raise NotImplementedError(f'This seed ({seed}) already exists for xy {point}.')
            # (set(range(len(x))) - set(x)).pop()
        return seed

    def ask(self, n, tell_pending=True):
        points, loss_improvements = super().ask(n, tell_pending=False)

        max_loss = max(loss_improvements)
        # '_stack' is {new_point_inside_triangle: loss_improvement, ...}
        # 'data_sem' is {existing_points: normalized_standard_error, ...}
        #  where 'normalized_standard_error = weight * standard_error / z_scale'.
        for (p, sem) in self.data_sem.items():
            if sem > 1:
                points.append((p, self.get_seed(p)))
                loss_improvements.append(sem)

        points, loss_improvements = zip(*sorted(zip(points, loss_improvements),
            key=operator.itemgetter(1), reverse=True))

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
        self.pending_points = defaultdict(set)
        for p in self._bounds_points:
            if p not in self.data:
                self._stack[(p, 0)] = np.inf
