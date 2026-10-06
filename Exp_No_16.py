from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(20,), (30, 15)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [400]
}

mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
test_acc = best_model.score(X_test, y_test)

print("Hyperparameter Tuning Results:")
print("Best Parameters:", grid_search.best_params_)
print(f"Best Validation Score: {grid_search.best_score_*100:.2f}%")
print(f"Test Set Accuracy: {test_acc*100:.2f}%")
