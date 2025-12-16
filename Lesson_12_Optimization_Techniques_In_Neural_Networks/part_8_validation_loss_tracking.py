validation_losses = []

for epoch in range(100):
    # Training loop here
    
    # Validation loss check
    with torch.no_grad():
        val_predictions = model(val_x)
        val_loss = criterion(val_predictions, val_y)
        validation_losses.append(val_loss.item())
        
    if val_loss < min(validation_losses):
        # Save best model
