# -*- coding: utf-8 -*-
from collections import defaultdict, OrderedDict
import functools
from operator import itemgetter

from .base_learner import BaseLearner
from .utils import restore


def dispatch(child_functions, arg):
    index, x = arg
    return child_functions[index](x)


class BalancingLearner(BaseLearner):
    """Choose the optimal points from a set of learners.

    Parameters
    ----------
    learners : sequence of BaseLearner
        The learners from which to choose. These must all have the same type.

    Notes
    -----
    This learner compares the 'loss' calculated from the "child" learners.
    This requires that the 'loss' from different learners *can be meaningfully
    compared*. For the moment we enforce this restriction by requiring that
    all learners are the same type but (depending on the internals of the
    learner) it may be that the loss cannot be compared *even between learners
    of the same type*. In this case the BalancingLearner will behave in an
    undefined way.
    """

    def __init__(self, learners):
        self.learners = learners

        # Naively we would make 'function' a method, but this causes problems
        # when using executors from 'concurrent.futures' because we have to
        # pickle the whole learner.
        self.function = functools.partial(dispatch, [l.function for l
                                                     in self.learners])

        self._points = defaultdict(OrderedDict)
        self._loss = defaultdict(OrderedDict)
        self.stack_size = 1

        if len(set(learner.__class__ for learner in self.learners)) > 1:
            raise TypeError('A BalacingLearner can handle only one type'
                            'of learners.')

    def _choose_and_add_points(self, n):
        points = []
        for _ in range(n):
            loss_improvements = []
            pairs = []
            for index, learner in enumerate(self.learners):
                if (index not in self._points
                    or (index in self._points
                        and len(self._points[index]) == 0)):
                    pls = learner.choose_points(
                        n=self.stack_size, add_data=False)
                    for point, loss_improvement in zip(*pls):
                        self._points[index][point] = loss_improvement

                point, loss_improvement = list(self._points[index].items())[0]
                loss_improvements.append(loss_improvement)
                pairs.append((index, point))
            x, _ = max(zip(pairs, loss_improvements), key=itemgetter(1))
            points.append(x)
            self.add_point(x, None)

        return points, None

    def choose_points(self, n, add_data=True):
        """Chose points for learners."""
        if not add_data:
            with restore(*self.learners):
                return self._choose_and_add_points(n)
        else:
            return self._choose_and_add_points(n)

    def add_point(self, x, y):
        index, x = x
        self._points[index].pop(x, None)
        self._loss.pop(index, None)
        self.learners[index].add_point(x, y)

    def loss(self, real=True):
        losses = []
        for index, learner in enumerate(self.learners):
            if index not in self._loss:
                self._loss[index] = learner.loss(real)
            loss = self._loss[index]
            losses.append(loss)
        return max(losses)

    def plot(self, index):
        return self.learners[index].plot()

    def remove_unfinished(self):
        """Remove uncomputed data from the learners."""
        for learner in self.learners:
            learner.remove_unfinished()
