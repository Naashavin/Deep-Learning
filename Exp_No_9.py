from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

architectures = [
    (10,),
    (30, 15),
    (50, 25, 10)
]
activations = ['relu', 'tanh']
learning_rates = [0.001, 0.01]

print(f"{'Layers':<15} | {'Activation':<10} | {'LR':<6} | {'Accuracy':<10}")
print("=" * 50)

for arch in architectures:
    for act in activations:
        for lr in learning_rates:
            mlp = MLPClassifier(hidden_layer_sizes=arch, activation=act, learning_rate_init=lr, max_iter=600, random_state=42)
            mlp.fit(X_train, y_train)
            acc = accuracy_score(y_test, mlp.predict(X_test))
            print(f"{str(arch):<15} | {act:<10} | {lr:<6} | {acc*100:.2f}%")
