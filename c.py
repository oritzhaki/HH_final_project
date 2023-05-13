import optuna

def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    num_layers = trial.suggest_int('num_layers', 1, 5)

    # Train and evaluate your model with the given hyperparameters
    # Return the performance metric you want to optimize

    # Example: Dummy objective function
    loss = (learning_rate - 0.05) ** 2 + num_layers

    return loss

study = optuna.create_study(direction='minimize')

# Optimize the hyperparameters
study.optimize(objective, n_trials=100)

# Retrieve the best hyperparameters
best_params = study.best_params
best_value = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Value:", best_value)
