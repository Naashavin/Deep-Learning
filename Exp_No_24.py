import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_spiral_data(n_points_per_class=250):
    np.random.seed(42)
    X = []
    y = []
    for class_number in range(2):
        r = np.linspace(0.0, 1, n_points_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, n_points_per_class) + np.random.randn(n_points_per_class) * 0.2
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_points_per_class, class_number))
    return np.vstack(X), np.hstack(y)

X, y = generate_spiral_data(250)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("NN Model Performance on Spiral Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
