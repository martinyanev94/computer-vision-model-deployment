def train(self, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        y_pred = self.feedforward(X)
        loss = mean_squared_error(y, y_pred)
        self.backpropagation(X, y, learning_rate)

        if epoch % 100 == 0:  # Logging every 100 epochs
            print(f'Epoch {epoch}, Loss: {loss}')
