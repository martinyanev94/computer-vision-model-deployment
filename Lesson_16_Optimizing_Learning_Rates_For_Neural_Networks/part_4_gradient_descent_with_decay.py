def exp_decay_learning_rate(initial_lr, decay_rate, iteration):
    return initial_lr * np.exp(-decay_rate * iteration)
def gradient_descent_with_schedule(X, y, initial_lr, decay_rate, n_iterations):
    m = X.shape[0]
    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((m, 1)), X]  # adding x0 = 1
    
    for iteration in range(n_iterations):
        lr = exp_decay_learning_rate(initial_lr, decay_rate, iteration)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    
    return theta
# Using a learning rate schedule
theta_with_schedule = gradient_descent_with_schedule(X, y, initial_lr=0.1, decay_rate=0.01, n_iterations=1000)
print("Parameters with learning rate schedule:", theta_with_schedule)
