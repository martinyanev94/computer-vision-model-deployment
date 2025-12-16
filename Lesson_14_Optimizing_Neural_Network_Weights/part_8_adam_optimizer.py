def adam_optimizer(weights, gradients, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for i in range(len(weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] ** 2)
        
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        
        weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

# Example initialization for Adam
m = [np.zeros(w.shape) for w in weights]
v = [np.zeros(w.shape) for w in weights]
t = 1  # Step count

# Use the optimizer during weight updates
