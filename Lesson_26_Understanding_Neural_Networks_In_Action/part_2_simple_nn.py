class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input_layer = nn.Linear(28 * 28, 128)  # Assuming input images are 28x28 pixels
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)  # 10 classes for output
        
    def forward(self, x):
        x = torch.relu(self.input_layer(x))  # Using ReLU as the activation function
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
