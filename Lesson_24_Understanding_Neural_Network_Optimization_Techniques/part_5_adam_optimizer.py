def adam_optimizer(weights, gradients, learning_rate, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * gradients
    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * gradients**2
    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)
    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)
    # Update weights
    weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return weights, m, v
