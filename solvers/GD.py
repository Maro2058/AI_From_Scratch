import numpy as np
from solvers.base_solver import Optimizer

class GradientDescent(Optimizer):
    def step(self, params, grads, eta = 0.1):
        params = params - eta * grads
        return params