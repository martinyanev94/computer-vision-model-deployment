def cyclical_learning_rate(base_lr, max_lr, step_size, iteration):
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))
def gradient_descent_with_cyclical_lr(X, y, base_lr, max_lr, step_size, n_iterations):
    m = X.shape[0]
    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((m, 1)), X]
    
    for iteration in range(n_iterations):
        lr = cyclical_learning_rate(base_lr, max_lr, step_size, iteration)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    
    return theta
# Using cyclic learning rates
theta_with_cyclic_lr = gradient_descent_with_cyclical_lr(X, y, base_lr=0.001, max_lr=0.1, step_size=200, n_iterations=1000)
print("Parameters with cyclic learning rate:", theta_with_cyclic_lr)
