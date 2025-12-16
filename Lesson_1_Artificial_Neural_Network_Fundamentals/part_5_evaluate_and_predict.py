test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')
import matplotlib.pyplot as plt

predictions = model.predict(x_test[:10])  # Get predictions for the first 10 test images
predicted_labels = tf.argmax(predictions, axis=1).numpy()

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Predicted: {predicted_labels[i]}')
    plt.axis('off')
plt.show()
