def stochastic_gradient_descent(weights, learning_rate, inputs, targets):
    for x, y in zip(inputs, targets):
        # Forward pass
        prediction = forward_pass(weights, x)
        loss = calculate_loss(prediction, y)

        # Backward pass
        gradients = compute_gradients(weights, loss)

        # Update weights
        weights = gradient_descent(weights, learning_rate, gradients)

    return weights
