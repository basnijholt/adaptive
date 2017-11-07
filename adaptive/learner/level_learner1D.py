# -*- coding: utf-8 -*-
import holoviews as hv
import numpy as np
import scipy.interpolate

from .learner1D import Learner1D

class LevelLearner1D(Learner1D):
    """Learns and predicts a function 'f:ℝ → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single real parameter and
        return an array of numbers.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    """
    def interval_loss(self, x_left, x_right, data):
        """Calculate loss in the interval x_left, x_right."""
        y_right, y_left = data[x_right], data[x_left]
        x_scale, y_scale = self._scale
        if y_scale == 0:
            loss = (x_right - x_left) / x_scale
        else:
            loss = np.hypot((x_right - x_left) / x_scale,
                            (y_right - y_left) / y_scale)
        return loss.max()

    def update_scale(self, x, y):
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        if y is not None:
            self._bbox[1][0] = min(self._bbox[1][0], min(y))
            self._bbox[1][1] = max(self._bbox[1][1], max(y))

        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

    def interpolate(self, extra_points=None):
        xs = list(self.data.keys())
        ys = np.array(list(self.data.values())).T
        xs_unfinished = list(self.data_interp.keys())

        if extra_points is not None:
            xs_unfinished += extra_points

        if len(xs) < 2:
            n_levels = max(1, ys.shape[0])
            interp_ys = np.zeros((n_levels, len(xs_unfinished)))
        else:
            ip = scipy.interpolate.interp1d(xs, ys,
                                            assume_sorted=True,
                                            bounds_error=False,
                                            fill_value=0)
            interp_ys = ip(xs_unfinished)

        data_interp = {x: y for x, y in zip(xs_unfinished, interp_ys.T)}

        return data_interp

    def plot(self):
        if self.data:
            xs = list(self.data.keys())
            ys = np.array(list(self.data.values())).T
            return hv.Overlay([hv.Scatter((xs, y)) for y in ys])
        else:
            return hv.Overlay([hv.Scatter([])])
