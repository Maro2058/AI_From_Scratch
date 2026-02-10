import numpy as np
import matplotlib.pyplot as plt
from models.Regression import LinearRegression
from solvers.GD import GradientDescent
from solvers.ClosedForm import SVD
from losses.metrics import MSE

# For testing and Demonstration purposes only

w0 = 1
w1 = 5
X = 2*np.random.rand(100,1)
y = w1*X + w0 + np.random.randn(100, 1)

linear = LinearRegression()
linear.compile(Solver = GradientDescent(), loss = MSE())
linear.train(X, y, 100000, 0.01)
y_pred = linear.predict(X)
print(y[1:5])
print("\nPredictions:")
print(y_pred[1:5])
error = MSE()
print(f"error: {error.value(y, y_pred)}")
linear.get_paramaters()

linear2 = LinearRegression()
linear2.compile(SVD())
linear2.train(X, y)
y_pred_2 = linear2.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, "r-", label = "Predictions")
plt.show()
