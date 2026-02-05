import numpy as np
import scipy.io as sio
from data.data_manipulation import add_dummy_variable
from models.base_model import model

class LinearRegression(model):
    def __init__(self, penalty ):
        self.weights = None
        self.intercept = None
        self.penalty = penalty

    def predict(self, X):
        return NotImplemented
    
    def get_paramaters(self):
        print(self.weights)
        print(self.intercept)
        
    
    def set_paramaters(self, weights = None, intercept = None):
        self.weights = weights
        self.intercept = intercept

class OldLinearRegression():

    # Consider adding Momentum later
    def __init__(self, penalty = None, lr = 0.01):
        self.weights = None
        self.penalty = penalty
        self.lr = lr

    def train(self, X, y):
        if self.penalty == "l1":
            self.train_lasso(X, y)
        elif self.penalty == "l2":
            self.train_ridge(X,y)
        elif self.penalty == "elastic":
            self.train_elastic(X, y)
        else:
            print("Linear Regression here")

    def train_least_squares(self, X, y):
        X_new = add_dummy_variable(X)
        self.weights = np.linalg.pinv(X_new) @ y
        return self.weights

    def predict(self, X):
        return
    def train_lasso(self, X, y):
        return
    def train_ridge(self, X, y):
        return
    def train_elastic(self, X, y):
        return

