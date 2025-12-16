import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for the first hidden layer
        x = torch.relu(self.fc2(x))  # Activation function for the second hidden layer
        x = self.fc3(x)  # Output layer
        return x
