import numpy as np
import scipy.io as sio

# General Use Functions
def add_dummy_variable(X, numberOfDummy = 1):
    instances , features = np.shape(X)
    new_columns = np.ones([instances, numberOfDummy])
    X_new = np.concatenate((X, new_columns), axis=1)
    return X_new

def polynomial_features(X, degree):
    return

def MSE(y, y_pred):
    return

def accuracy(y, y_pred):
    return

def train_test_split(X, y, shuffle = False):
    if shuffle == True:
        return
    # Shuffle Data
    # 80-20 split
    return

def train_val_test_split(X, y, shuffle = False):
    if shuffle == True:
        return
    # Shuffle Data
    # 80-20 split
    return

class LinearRegression():

    # Consider adding Momentum later
    def __init__(self, penalty = None, lr = 0.01):
        self.weights = None
        self.intercept = None
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

w0 = 1
w1 = 3
w2 = 5
X = 2*np.random.rand(100,1)
y = w2*X + w0 + np.random.randn(100, 1)
linear = LinearRegression()
print(linear.train_least_squares(X, y))