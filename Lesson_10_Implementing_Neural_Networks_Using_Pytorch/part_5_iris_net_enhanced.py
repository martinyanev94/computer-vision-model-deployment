class IrisNetEnhanced(nn.Module):
    def __init__(self):
        super(IrisNetEnhanced, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # First hidden layer with 8 neurons
        self.fc2 = nn.Linear(8, 4)   # Second hidden layer with 4 neurons
        self.fc3 = nn.Linear(4, 3)    # Output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Instantiate the enhanced model
model_enhanced = IrisNetEnhanced()
