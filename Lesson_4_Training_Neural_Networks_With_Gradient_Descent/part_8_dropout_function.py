def dropout(self, layer_output, dropout_rate):
    if dropout_rate < 1.0:
        mask = np.random.binomial(1, 1 - dropout_rate, size=layer_output.shape)
        return layer_output * mask / (1 - dropout_rate)  # Scale up the output
    return layer_output
