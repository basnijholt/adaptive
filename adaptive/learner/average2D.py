# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
import itertools
from math import sqrt
import sys

import numpy as np

from .learner2D import Learner2D, default_loss, choose_point_in_triangle, areas


def unpack_point(point):
    return tuple(point[0]), point[1]


def standard_error(lst):
    n = len(lst)
    if n < 3:
        return sys.float_info.max
    sum_f_sq = sum(x**2 for x in lst)
    mean = sum(x for x in lst) / n
    try:
        std = sqrt((sum_f_sq - n * mean**2) / (n - 1))
    except ValueError:  # sum_f_sq - n * mean**2 is numerically 0
        return 0
    return std / sqrt(n)


class AverageLearner2D(Learner2D):
    def __init__(self, function, bounds, loss_per_triangle=None):
        super().__init__(function, bounds, loss_per_triangle)
        self._data = defaultdict(lambda: defaultdict(dict))
        self.pending_points = defaultdict(set)

        # Adding a seed of 0 to the _stack to
        # make {((x, y), seed): loss_improvements, ...}.
        self._stack = {(p, 0): l for p, l in self._stack.items()}

    @property
    def bounds_are_done(self):
        return all(p in self.data for p in self._bounds_points)

    @property
    def data(self):
        return {k: sum(v.values()) / len(v) for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: standard_error(v.values()) for k, v in self._data.items()}

    def _add_to_pending(self, point):
        xy, seed = unpack_point(point)
        self.pending_points[xy].add(seed)

    def _remove_from_to_pending(self, point):
        xy, seed = unpack_point(point)
        self.pending_points[xy].discard(seed)

    def _add_to_data(self, point, value):
        xy, seed = unpack_point(point)
        self._data[xy][seed] = value

    def _points_and_loss_improvements_from_stack(self):
        if len(self._stack) < 1:
            self._fill_stack(self.stack_size)

        # '_stack' is {new_point_inside_triangle: loss_improvement, ...}
        # 'data_sem' is {existing_points: standard_error, ...}
        data_sem = {(p, len(values)): sem for (p, sem), values in
                    zip(self.data_sem.items(), self._data.values())}
        # stack = {((x, y), seed): loss_improvements_or_standard_error}
        stack_with_seed = {**self._stack, **data_sem}

        points, loss_improvements = map(list,
            zip(*sorted(stack_with_seed.items(), key=lambda x: -x[1]))
        )
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
