import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):

    capabilities = {}
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.is_trained = False
        self.hyperparameters = {}

    # Forward Pass: w*X
    @abstractmethod
    def predict(self, X : np.ndarray):
        raise NotImplementedError
    
    # Backward Pass: dL/dw = dL/dy_pred * dy_pred/dw
    def gradient(self, dL_dypred):
        pass
    
    @abstractmethod
    def get_paramaters(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_paramaters(self, weights: np.ndarray):
        raise NotImplementedError

# Class functions usually need either model and X or simply just y_pred, depending on how much each class needs to know.
# Solver uses both model and X instead of just y_pred, becuase one way or another it still needs the model to fit it to X.

