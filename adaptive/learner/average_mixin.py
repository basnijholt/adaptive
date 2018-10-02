# -*- coding: utf-8 -*-

from math import sqrt
import sys

import numpy as np

from .learner1D import Learner1D


class AverageMixin:
    @property
    def data(self):
        return {k: sum(v.values()) / len(v) for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: self.standard_error(v.values()) for k, v in self._data.items()}

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
        scale = self.value_scale()

        points = []
        loss_improvements = []
        for p, sem in self.data_sem.items():
            if sem > 0:
                points.append((p, self.get_seed(p)))
                N = len(self._data[p] + len(self.pending_points.get(p, [])))
                sem_improvement = (1 - sqrt(N - 1) / sqrt(N)) * sem
                loss_improvements.append(self.weight * sem_improvement / scale)
        return points, loss_improvements

    def _add_to_pending(self, point):
        x, seed = self.unpack_point(point)
        if x not in self.pending_points:
            self.pending_points[x] = set()
        self.pending_points[x].add(seed)

    def _remove_from_to_pending(self, point):
        x, seed = self.unpack_point(point)
        if x in self.pending_points:
            self.pending_points[x].discard(seed)
            if not self.pending_points[x]:
                # pending_points[x] is now empty so delete the set()
                del self.pending_points[x]

    def _add_to_data(self, point, value):
        x, seed = self.unpack_point(point)
        if x not in self._data:
            self._data[x] = {}
        self._data[x][seed] = value

    def ask(self, n, tell_pending=True):
        """Return n points that are expected to maximally reduce the loss."""
        points, loss_improvements = self._ask_points_without_adding(n)
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


def add_average_mixin(cls):
    names = ('data', 'data_sem', 'standard_error', 'mean_values_per_point',
             'get_seed', 'loss_per_existing_point', '_add_to_pending',
             '_remove_from_to_pending', '_add_to_data', 'ask')

    for name in names:
        setattr(cls, name, getattr(AverageMixin, name))

    return cls
