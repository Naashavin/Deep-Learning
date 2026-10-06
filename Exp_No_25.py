from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

activations = ['relu', 'tanh', 'logistic']
structures = [
    (16,),
    (32, 16),
    (64, 32, 16)
]

print(f"{'Structure':<18} | {'Activation':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
print("=" * 55)

for struct in structures:
    for act in activations:
        mlp = MLPClassifier(hidden_layer_sizes=struct, activation=act, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        
        tr_acc = accuracy_score(y_train, mlp.predict(X_train))
        ts_acc = accuracy_score(y_test, mlp.predict(X_test))
        
        print(f"{str(struct):<18} | {act:<10} | {tr_acc*100:.2f}%     | {ts_acc*100:.2f}%")
