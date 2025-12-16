import optuna

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    
    # Model definition with suggested dropout
    model = AdvancedNN(input_size, hidden_size1, hidden_size2, output_size, dropout_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop goes here...
    
    return loss.item()  # Return the loss as the objective metric

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
