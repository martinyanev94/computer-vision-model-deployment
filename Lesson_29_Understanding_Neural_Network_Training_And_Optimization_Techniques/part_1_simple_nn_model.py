import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Input layer to hidden layer
        self.fc2 = nn.Linear(5, 1)    # Hidden layer to output layer
        self.activation = nn.ReLU()    # Activation function

    def forward(self, x):
        x = self.fc1(x)   # Linear transformation
        x = self.activation(x)  # Apply activation
        x = self.fc2(x)   # Another linear transformation
        return x  # Final output

model = SimpleNN()
input_data = torch.randn(1, 10)  # Generate random input data
output = model(input_data)  # Perform forward propagation
print(output)
