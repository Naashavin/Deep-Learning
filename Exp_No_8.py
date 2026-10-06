from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

learning_rates = [0.001, 0.01, 0.1]
activations = ['relu', 'tanh', 'logistic']

print(f"{'Activation':<12} | {'Learning Rate':<15} | {'Test Accuracy':<12}")
print("-" * 45)

for act in activations:
    for lr in learning_rates:
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation=act, learning_rate_init=lr, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy_score(y_test, mlp.predict(X_test))
        print(f"{act:<12} | {lr:<15} | {acc * 100:.2f}%")
