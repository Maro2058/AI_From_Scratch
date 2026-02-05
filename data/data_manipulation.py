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