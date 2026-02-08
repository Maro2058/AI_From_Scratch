from abc import ABC, abstractmethod
import numpy as np
from models.base_model import Model
from solvers.base_solver import Optimizer, Estimator
from losses.base_loss import Loss

class Trainer(ABC):
    @abstractmethod
    def train(self, model, X, y):
        pass


# Uses Iterative solver (Optimizers)
class IterativeTrainer(Trainer):

    def __init__(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def train(self,
              model: Model,
              X: np.ndarray,
              y: np.ndarray,
              epochs : int = 1,
              batch_size=None,
              shuffle = True,
              metrics=[]):

        pass

# Uses Closed form solvers (Estimators)
class ClosedFromTrainer(Trainer):
    
    def __init__(self, estimator : Estimator):
        self.estimator = estimator

    def train(self, model : Model, X, y):
        model.weights = self.estimator.solve(X, y)
        pass