import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
# Create some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # true relationship: y = 4 + 3x + noise
def gradient_descent(X, y, learning_rate, n_iterations):
    m = X.shape[0]
    theta = np.random.randn(2, 1)  # random initialization
    X_b = np.c_[np.ones((m, 1)), X]  # add x0 = 1 to each instance

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients
    
    return theta
# Run gradient descent with different learning rates
high_lr_theta = gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)
low_lr_theta = gradient_descent(X, y, learning_rate=0.001, n_iterations=1000)

print("Parameters with high learning rate:", high_lr_theta)
print("Parameters with low learning rate:", low_lr_theta)
