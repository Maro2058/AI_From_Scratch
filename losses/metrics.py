import numpy as np
import scipy as sio
from losses.base_loss import Loss

class MSE(Loss):
    def value(self, y, y_pred):
        m = y.shape[0]
        error = y - y_pred
        MSE = (error**2).sum() / m
        return MSE
    
    def backward(self, y, y_pred):
        m = y.shape[0]
        dL_dy = (2/m) * (y_pred - y)
        return dL_dy



def accuracy(y, y_pred):
    return

def true_pos_rate(y, y_pred):
    return

def true_neg_rate(y, y_pred):
    return

def false_pos_rate(y, y_pred):
    return

def false_neg_rate(y, y_pred):
    return

def recall(y, y_pred):
    return

def precision(y, y_pred):
    return

def confusion_matrix(y, y_pred):
    return

def F1_Score(y, y_pred):
    return

def ROC(y, y_pred):
    return

def ROC_AUC(y, y_pred):
    return