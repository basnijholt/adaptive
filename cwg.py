import adaptive
import numpy as np
from adaptive.learner import IntegratorLearner
from adaptive.tests.algorithm_4 import algorithm_4, f0

def run_integrator_learner(f, a, b, tol, nr_points):
    learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
    for _ in range(nr_points):
        points, _ = learner.choose_points(1)
        learner.add_data(points, map(learner.function, points))
    return learner

def same_ivals_up_to(f, a, b, tol, N=1):
    igral, err, nr_points, ivals = algorithm_4(f, a, b, tol, N+1)

    learner = run_integrator_learner(f, a, b, tol, nr_points)

    print('igral difference', learner.igral-igral,
          'err difference', learner.err - err)

    return learner.equal(ivals, verbose=True)

f, a, b, tol = f0, 0, 3, 1e-5
igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)
learner = run_integrator_learner(f, a, b, tol, nr_points)
learner.equal(ivals, verbose=True)

