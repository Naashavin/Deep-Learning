import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

raw_preds = model.predict(X_test)
binary_preds = np.where(raw_preds >= 0.5, 1, 0)

print("Linear Regression as Classifier (Breast Cancer):")
print("Accuracy:", accuracy_score(y_test, binary_preds))
print("\nClassification Report:\n", classification_report(y_test, binary_preds))
