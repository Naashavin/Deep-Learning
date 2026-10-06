from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

overfitted_model = DecisionTreeClassifier(max_depth=None, random_state=42)
overfitted_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, overfitted_model.predict(X_train))
test_acc = accuracy_score(y_test, overfitted_model.predict(X_test))

print("Unconstrained Decision Tree (Overfitted):")
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy:  {test_acc * 100:.2f}%\n")

regularized_model = DecisionTreeClassifier(max_depth=3, random_state=42)
regularized_model.fit(X_train, y_train)

reg_train_acc = accuracy_score(y_train, regularized_model.predict(X_train))
reg_test_acc = accuracy_score(y_test, regularized_model.predict(X_test))

print("Constrained Decision Tree (Regularized):")
print(f"Train Accuracy: {reg_train_acc * 100:.2f}%")
print(f"Test Accuracy:  {reg_test_acc * 100:.2f}%")
