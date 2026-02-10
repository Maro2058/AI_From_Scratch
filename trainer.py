from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from solvers.base_solver import Optimizer, Estimator
from losses.base_loss import Loss

if TYPE_CHECKING:
    from models.base_model import Model


class Trainer(ABC):
    @abstractmethod
    def train(self, model : "Model", X, y):
        pass


# Uses Iterative solver (Optimizers)
class IterativeTrainer(Trainer):

    def __init__(self, optimizer : Optimizer, loss : Loss):
        self.optimizer = optimizer
        self.loss = loss

    def train(self,
              model: "Model",
              X: np.ndarray,
              y: np.ndarray,
              epochs : int = 10,
              eta = 0.1,
              batch_size=None,
              shuffle = True,
              metrics=[]):
        
        # Start of trainer logic:
        for epoch in range(epochs):
            y_pred, cache = model._forward(X)
            dL_dy = self.loss.backward(y, y_pred)
            gradients = model._gradient(dL_dy, cache)
            weights = self.optimizer.step(model.weights, gradients, eta)
            model.weights = weights


# Uses Closed form solvers (Estimators)
class ClosedFromTrainer(Trainer):
    
    def __init__(self, estimator : Estimator):
        self.estimator = estimator

    def train(self, model : "Model", X, y):
        model.weights = self.estimator.solve(X, y)
        pass