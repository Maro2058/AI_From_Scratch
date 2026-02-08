import numpy as np
from models.base_model import Model
from losses.base_loss import Loss
from abc import ABC, abstractmethod

class Optimizer(ABC):
    required_capabilities = {}

    def __init__(self, params, lr=0.01):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
    @abstractmethod
    def step(self, params, grads):
        raise NotImplementedError

class Estimator(ABC):
    required_capabilities = {}

    @abstractmethod
    def solve(self, X, y):
        pass