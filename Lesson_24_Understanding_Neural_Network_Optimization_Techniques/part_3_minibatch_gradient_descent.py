def minibatch_gradient_descent(weights, learning_rate, inputs, targets, batch_size):
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]

        # Forward pass for the minibatch
        predictions = forward_pass(weights, batch_inputs)
        losses = calculate_loss(predictions, batch_targets)

        # Backward pass for the minibatch
        gradients = compute_gradients(weights, losses)

        # Update weights
        weights = gradient_descent(weights, learning_rate, gradients)

    return weights
