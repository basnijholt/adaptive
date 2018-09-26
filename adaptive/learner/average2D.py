# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
import itertools
from math import sqrt
import sys

import numpy as np

from .learner2D import Learner2D, default_loss, choose_point_in_triangle, areas


def standard_error(lst):
    n = len(lst)
    if n < 2:
        return sys.float_info.max
    sum_f_sq = sum(x**2 for x in lst)
    mean = sum(x for x in lst) / n
    std = sqrt((sum_f_sq - n * mean**2) / (n - 1))
    return std / sqrt(n)


class AverageLearner2D(Learner2D):
    def __init__(self, function, bounds, loss_per_triangle=None):
        super().__init__(function, bounds, loss_per_triangle)
        self._data = defaultdict(list)  # only difference with Learner2D

    @property
    def data(self):
        return {k: sum(v) / len(v) for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: standard_error(v) for k, v in self._data.items()}

    def tell(self, point, value):
        point = tuple(point)
        self._data[point].append(value)  # only difference with Learner2D
        if not self.inside_bounds(point):
            return
        self.pending_points.discard(point)
        self._ip = None
        self._stack.pop(point, None)

    def _points_and_loss_improvements_from_stack(self):
        if len(self._stack) < 1:
            self._fill_stack(self.stack_size)

        # '_stack' is {new_point_inside_triangle: loss_improvement, ...}
        # 'data_sem' is {existing_points: standard_error, ...}
        stack = {**self._stack, **self.data_sem}

        points, loss_improvements = map(list,
            zip(*sorted(stack.items(), key=lambda x: -x[1]))
        )
        return points, loss_improvements
