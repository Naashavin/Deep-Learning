from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
pred_linear = svm_linear.predict(X_test)

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
pred_rbf = svm_rbf.predict(X_test)

print("Support Vector Machine Classifier (Iris Dataset):")
print("Linear Kernel Accuracy:", accuracy_score(y_test, pred_linear))
print("RBF Kernel Accuracy:   ", accuracy_score(y_test, pred_rbf))
print("\nRBF Kernel Classification Report:\n", classification_report(y_test, pred_rbf))
