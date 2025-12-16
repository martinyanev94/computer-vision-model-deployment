class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer after convolution
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First convolution followed by ReLU
        x = F.max_pool2d(x, 2)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Second convolution followed by ReLU
        x = F.max_pool2d(x, 2)  # Apply second max pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected layer
        return F.log_softmax(self.fc2(x), dim=1)  # Output layer

cnn_model = CNN()
print(cnn_model)
