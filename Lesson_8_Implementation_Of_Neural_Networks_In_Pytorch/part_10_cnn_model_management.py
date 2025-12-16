torch.save(cnn_model.state_dict(), 'cnn_model.pth')  # Save the model
cnn_model = CNN()  # Re-instantiate the model
cnn_model.load_state_dict(torch.load('cnn_model.pth'))  # Load the weights
cnn_model.eval()  # Set to evaluation mode
