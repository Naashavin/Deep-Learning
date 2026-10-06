import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred_binary = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
tn, fp, fn, tp = cm_binary.ravel()

print("--- Binary Confusion Matrix ---")
print(cm_binary)
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
print(f"Accuracy: {accuracy_score(y_true_binary, y_pred_binary):.2f}")
print(f"Precision: {precision_score(y_true_binary, y_pred_binary):.2f}")
print(f"Recall: {recall_score(y_true_binary, y_pred_binary):.2f}")
print(f"F1-Score: {f1_score(y_true_binary, y_pred_binary):.2f}\n")

y_true_multi = [0, 1, 2, 0, 1, 2, 0, 2, 1, 2]
y_pred_multi = [0, 1, 1, 0, 1, 2, 0, 1, 2, 2]

cm_multi = confusion_matrix(y_true_multi, y_pred_multi)
print("--- Multi-class Confusion Matrix ---")
print(cm_multi)
print(f"Accuracy: {accuracy_score(y_true_multi, y_pred_multi):.2f}")
print(f"Precision (macro): {precision_score(y_true_multi, y_pred_multi, average='macro'):.2f}")
print(f"Recall (macro): {recall_score(y_true_multi, y_pred_multi, average='macro'):.2f}")
print(f"F1-Score (macro): {f1_score(y_true_multi, y_pred_multi, average='macro'):.2f}")
