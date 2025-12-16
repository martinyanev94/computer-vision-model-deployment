class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)  # Adding dropout layer
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden_out = self.activation(self.hidden(x))
        hidden_out = self.dropout(hidden_out)  # Applying dropout
        output_out = self.output(hidden_out)
        return output_out
