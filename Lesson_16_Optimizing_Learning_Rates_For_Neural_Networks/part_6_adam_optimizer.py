def adam_optimizer(X, y, n_iterations, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = X.shape[0]
    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((m, 1)), X]
    
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    t = 0
    
    for iteration in range(n_iterations):
        t += 1
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        
        m_t = beta1 * m_t + (1 - beta1) * gradients
        v_t = beta2 * v_t + (1 - beta2) * gradients**2
        
        m_hat = m_t / (1 - beta1**t)
        v_hat = v_t / (1 - beta2**t)
        
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return theta
# Using Adam optimizer
theta_with_adam = adam_optimizer(X, y, n_iterations=1000)
print("Parameters with Adam optimizer:", theta_with_adam)
