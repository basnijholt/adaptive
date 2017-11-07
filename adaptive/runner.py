# -*- coding: utf-8 -*-
import asyncio
import concurrent.futures as concurrent
from collections import defaultdict
import time

import ipyparallel


class Runner:
    """Runs a learning algorithm in an executor.

    Parameters
    ----------
    learner : Learner
    executor : concurrent.futures.Executor, or ipyparallel.Client, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new ProcessPoolExecutor is used.
    goal : callable, optional
        The end condition for the calculation. This function must take the
        learner as its sole argument, and return True if we should stop.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner
    ioloop : asyncio.AbstractEventLoop, optional
        The ioloop in which to run the learning algorithm. If not provided,
        the default event loop is used.
    shutdown_executor : Bool, default: True
        If True, shutdown the executor when the runner has completed. If
        'executor' is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.

    Attributes
    ----------
    task : asyncio.Task
        The underlying task. May be cancelled to stop the runner.
    learner : Learner
        The underlying learner. May be queried for its state
    log : list or None
        Record of the method calls made to the learner, in the format
        '(method_name, *args)'.
    """

    def __init__(self, learner, executor=None, goal=None, *,
                 log=False, ioloop=None, shutdown_executor=True,
                 timeit=False):
        self.ioloop = ioloop if ioloop else asyncio.get_event_loop()
        # if we instantiate our own executor, then we are also responsible
        # for calling 'shutdown'
        self.shutdown_executor = shutdown_executor or (executor is None)
        self.executor = ensure_async_executor(executor, self.ioloop)
        self.learner = learner
        self.log = [] if log else None
        self.timeit = timeit

        if goal is None:
            def goal(_):
                return False

        self.goal = goal

        coro = self._run()
        self.task = self.ioloop.create_task(coro)

        self.add_point = self.learner.add_point
        self.choose_points = self.learner.choose_points
        self.function = self.learner.function

        self.times = defaultdict(list)
        if self.timeit:
            self.add_point = self.time(self.add_point,
                                       key='add_point')
            self.choose_points = self.time(self.choose_points,
                                           key='choose_points')
            self.function = self.time(self.function,
                                      key='function')

    def run_sync(self):
        return self.ioloop.run_until_complete(self.task)

    async def _run(self):
        first_completed = asyncio.FIRST_COMPLETED
        xs = dict()
        done = [None] * self.executor.ncores
        do_log = self.log is not None

        if len(done) == 0:
            raise RuntimeError('Executor has no workers')

        try:
            while not self.goal(self.learner):
                # Launch tasks to replace the ones that completed
                # on the last iteration.
                if do_log:
                    self.log.append(('choose_points', len(done)))

                points, _ = self.choose_points(len(done))
                for x in points:
                    xs[self.executor.submit(self.function, x)] = x

                # Collect and results and add them to the learner
                futures = list(xs.keys())
                done, _ = await asyncio.wait(futures,
                                             return_when=first_completed,
                                             loop=self.ioloop)
                for fut in done:
                    x = xs.pop(fut)
                    y = await fut
                    if do_log:
                        self.log.append(('add_point', x, y))
                    self.add_point(x, y)
        finally:
            # remove points with 'None' values from the learner
            self.learner.remove_unfinished()
            # cancel any outstanding tasks
            remaining = list(xs.keys())
            if remaining:
                for fut in remaining:
                    fut.cancel()
                await asyncio.wait(remaining)
            if self.shutdown_executor:
                self.executor.shutdown()

    def time(self, method, key):
        def timed(*args, **kwargs):
            t_start = time.time()
            result = method(*args, **kwargs)
            t_end = time.time()
            self.times[key].append(1000 * (t_end - t_start))
            return result
        return timed

def replay_log(learner, log):
    """Apply a sequence of method calls to a learner.

    This is useful for debugging runners.

    Parameters
    ----------
    learner : learner.BaseLearner
    log : list
        contains tuples: '(method_name, *args)'.
    """
    for method, *args in log:
        getattr(learner, method)(*args)


def ensure_async_executor(executor, ioloop):
    if executor is None:
        executor = concurrent.ProcessPoolExecutor()
    elif isinstance(executor, concurrent.Executor):
        pass
    elif isinstance(executor, ipyparallel.Client):
        executor = executor.executor()
    else:
        raise TypeError('Only concurrent.futures.Executors or ipyparallel '
                        'clients can be used.')

    return _AsyncExecutor(executor, ioloop)


class SequentialExecutor(concurrent.Executor):
    """A trivial executor that runs functions synchronously.

    This executor is mainly for testing.
    """
    def submit(self, fn, *args, **kwargs):
        fut = concurrent.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterable, timeout=None, chunksize=1):
        return map(fn, iterable)

    def shutdown(self, wait=True):
        pass


# Internal functionality

class _AsyncExecutor:

    def __init__(self, executor, ioloop):
        assert isinstance(executor, concurrent.Executor)
        self.executor = executor
        self.ioloop = ioloop

    def submit(self, f, *args, **kwargs):
        return self.ioloop.run_in_executor(self.executor, f, *args, **kwargs)

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)

    @property
    def ncores(self):
        ex = self.executor
        if isinstance(ex, ipyparallel.client.view.ViewExecutor):
            return len(ex.view)
        elif isinstance(ex, (concurrent.ProcessPoolExecutor,
                             concurrent.ThreadPoolExecutor)):
            return ex._max_workers  # not public API!
        elif isinstance(ex, SequentialExecutor):
            return 1
        else:
            raise TypeError('Cannot get number of cores for {}'
                            .format(ex.__class__))
