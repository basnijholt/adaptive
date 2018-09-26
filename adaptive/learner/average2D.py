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

    def _fill_stack(self, stack_till=1):
        if len(self.data) + len(self.pending_points) < self.ndim + 1:
            raise ValueError("too few points...")

        # Interpolate
        ip = self.ip_combined()

        losses = self.loss_per_triangle(ip)

        points_new = []
        losses_new = []
        for j, _ in enumerate(losses):
            jsimplex = np.argmax(losses)
            triangle = ip.tri.points[ip.tri.vertices[jsimplex]]
            point_new = choose_point_in_triangle(triangle, max_badness=5)
            point_new = tuple(self._unscale(point_new))
            loss_new = abs(losses[jsimplex])  # only difference with Learner2D

            points_new.append(point_new)
            losses_new.append(loss_new)

            self._stack[point_new] = loss_new

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = -np.inf

        return points_new, losses_new

    def ask(self, n, tell_pending=True):
        # Even if tell_pending is False we add the point such that _fill_stack
        # will return new points, later we remove these points if needed.
        if len(self._stack) < 1:
            self._fill_stack(self.stack_size)

        # '_stack' is {new_point_inside_triangle: loss_improvement, ...}
        # 'data_sem' is {existing_points: standard_error, ...}
        stack = {**self._stack, **self.data_sem}

        points, loss_improvements = map(list,
            zip(*sorted(stack.items(), key=lambda x: -x[1]))
        )

        n_left = n - len(points)
        for p in points[:n]:
            self.tell_pending(p)

        while n_left > 0:
            # The while loop is needed because `stack_till` could be larger
            # than the number of triangles between the points. Therefore
            # it could fill up till a length smaller than `stack_till`.
            new_points, new_loss_improvements = self._fill_stack(
                stack_till=max(n_left, self.stack_size))
            for p in new_points[:n_left]:
                self.tell_pending(p)
            n_left -= len(new_points)

            points += new_points
            loss_improvements += new_loss_improvements

        if not tell_pending:
            self._stack = OrderedDict(zip(points[:self.stack_size],
                                          loss_improvements))
            for point in points[:n]:
                self.pending_points.discard(point)

        return points[:n], loss_improvements[:n]
