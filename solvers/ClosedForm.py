import numpy as np
from solvers.base_solver import Estimator

class SVD(Estimator):
    def solve(self, X, y):
        weights = np.linalg.pinv(X) @ y
        return weights