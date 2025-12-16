def train(self, X, y, epochs, learning_rate):
    loss_history = []
    for epoch in range(epochs):
        y_pred = self.feedforward(X)
        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)
        self.backpropagation(X, y, learning_rate)

        if epoch % 100 == 0:  # Log every 100 epochs
            print(f'Epoch {epoch}, Loss: {loss}')

    return loss_history
