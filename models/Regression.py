import numpy as np
import scipy.io as sio
from data.data_manipulation import add_dummy_variable
from models.base_model import Model
from trainer import Trainer, IterativeTrainer, ClosedFromTrainer
from solvers.ClosedForm import SVD
from solvers.base_solver import Solver, Optimizer, Estimator
from losses.base_loss import Loss

class LinearRegression(Model):
    def __init__(self, penalty = None):
        super().__init__()
        self.penalty = penalty

    def compile(self, Solver : Solver = SVD(), loss : Loss = None, metrics = []):
        if isinstance(Solver, Optimizer) and loss is None:
            raise ValueError(f"Must specify a loss function for optimizer {Solver}")
        return super().compile(Solver, loss, metrics)
    
    def get_paramaters(self):
        print(self.weights)
         
    def set_paramaters(self, weights = None, intercept = None):
        self.weights = weights
        self.intercept = intercept

    def _forward(self, X):
        cache = {"X" : X}
        y_pred = X @ self.weights
        return y_pred, cache

    def _gradient(self, dL_dy, cache):
        X = cache["X"]
        return X.T @ dL_dy

    def train(self, X: np.ndarray, y: np.ndarray, epochs = 10, eta = 0.1, batch_size = None, shuffle = True):

        X = add_dummy_variable(X)

        if self.solver is None or (isinstance(self.Solver, Optimizer) and self.loss is None):
            raise ValueError("Solver or loss is not set. Please compile the model before training.")

        if self.weights is None:
            no_of_features = X.shape[1]
            self.weights = np.zeros((no_of_features, 1))

        if isinstance(self.solver, Optimizer):
            trainer = IterativeTrainer(self.solver, self.loss)
            trainer.train(self, X, y, epochs, eta, batch_size, shuffle, self.metrics)
        elif isinstance(self.solver, Estimator):
            trainer = ClosedFromTrainer(self.solver)
            trainer.train(self, X, y)
        else:
            raise TypeError("Unkown Solver Type")
        
        self.is_trained = True

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Prediction failed. Weights are None. Train model first")
        X_b = add_dummy_variable(X)
        y_pred = X_b @ self.weights
        return y_pred

