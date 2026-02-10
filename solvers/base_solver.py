import numpy as np
from abc import ABC, abstractmethod


class Solver(ABC):
    pass

class Optimizer(Solver):
    required_capabilities = {}
        
    @abstractmethod
    def step(self, params, grads, eta):
        raise NotImplementedError

class Estimator(Solver):
    required_capabilities = {}

    @abstractmethod
    def solve(self, X, y):
        pass