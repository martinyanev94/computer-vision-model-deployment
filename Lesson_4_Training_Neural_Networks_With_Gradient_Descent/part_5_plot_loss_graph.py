import matplotlib.pyplot as plt

def plot_loss(loss_history):
    plt.plot(loss_history, color='blue')
    plt.title('Loss Reduction Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
