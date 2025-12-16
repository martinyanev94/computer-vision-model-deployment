class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)        # Hidden layer
        self.fc3 = nn.Linear(64, 10)         # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))  # Activation function for the first layer
        x = F.relu(self.fc2(x))  # Activation function for the second layer
        x = self.fc3(x)          # Output layer
        return F.log_softmax(x, dim=1)  # Softmax for multi-class classification

model = SimpleNN()
print(model)
criterion = nn.NLLLoss()  # Negative log likelihood loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
