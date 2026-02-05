class model():
    capabilities = {}

    def predict(self, X):
        raise NotImplementedError
    
    def get_paramaters(self):
        raise NotImplementedError
    
    def set_paramaters(self):
        raise NotImplementedError

# Class functions usually need either model and X or simply just y_pred, depending on how much each class needs to know.
# Solver uses both model and X instead of just y_pred, becuase one way or another it still needs the model to fit it to X.

