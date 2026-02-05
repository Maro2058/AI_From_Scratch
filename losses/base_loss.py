class loss():
    def value(self, y, y_pred):
        raise NotImplementedError
    
    def gradient(self, y, y_pred, X):
        raise NotImplementedError
