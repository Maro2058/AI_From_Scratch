class trainer():
    def train(self, model, solver, X, y, epochs, loss=None, metrics=[]):
        for iteration in range(epochs):
            solver.fit(model, X, y, loss)
        return