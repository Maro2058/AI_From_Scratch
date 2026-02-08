from abc import ABC, abstractmethod

class Loss(ABC):
    #forward pass: L
    @abstractmethod
    def value(self, y, y_pred):
        raise NotImplementedError
    
    #backward pass: dL/dy_pred
    @abstractmethod
    def backward(self, y, y_pred):
        raise NotImplementedError
