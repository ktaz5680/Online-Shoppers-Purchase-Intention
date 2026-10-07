import runpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# Run the modeling script and collect its variables
model = runpy.run_path("Scripts/04_modeling.py")

y_test = model["y_test"]

y_pred_baseline = model["y_pred_baseline"]
y_prob_baseline = model["y_prob_baseline"]

y_pred_only_eng = model["y_pred_only_eng"]
y_prob_only_eng = model["y_prob_only_eng"]

y_pred_eng = model["y_pred_eng"]
y_prob_eng = model["y_prob_eng"]


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Original Features",
        "Engineered Features Only",
        "Original + Engineered Features"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_only_eng),
        accuracy_score(y_test, y_pred_eng)
    ],

    "Precision": [
        precision_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_only_eng),
        precision_score(y_test, y_pred_eng)
    ],

    "Recall": [
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_only_eng),
        recall_score(y_test, y_pred_eng)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_only_eng),
        f1_score(y_test, y_pred_eng)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_baseline),
        roc_auc_score(y_test, y_prob_only_eng),
        roc_auc_score(y_test, y_prob_eng)
    ],

    "PR-AUC": [
        average_precision_score(y_test, y_prob_baseline),
        average_precision_score(y_test, y_prob_only_eng),
        average_precision_score(y_test, y_prob_eng)
    ]
})

print("\nModel Performance Comparison")
print(results.round(3))


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr_base, tpr_base, _ = roc_curve(
    y_test, y_prob_baseline
)

fpr_only, tpr_only, _ = roc_curve(
    y_test, y_prob_only_eng
)

fpr_eng, tpr_eng, _ = roc_curve(
    y_test, y_prob_eng
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr_base,
    tpr_base,
    label=f"Original Features (AUC = {roc_auc_score(y_test, y_prob_baseline):.3f})"
)

plt.plot(
    fpr_only,
    tpr_only,
    label=f"Engineered Features Only (AUC = {roc_auc_score(y_test, y_prob_only_eng):.3f})"
)

plt.plot(
    fpr_eng,
    tpr_eng,
    label=f"Original + Engineered (AUC = {roc_auc_score(y_test, y_prob_eng):.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Precision-Recall Curve
# --------------------------------------------------

precision_base, recall_base, _ = precision_recall_curve(
    y_test, y_prob_baseline
)

precision_only, recall_only, _ = precision_recall_curve(
    y_test, y_prob_only_eng
)

precision_eng, recall_eng, _ = precision_recall_curve(
    y_test, y_prob_eng
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_base,
    precision_base,
    label=f"Original Features (PR-AUC = {average_precision_score(y_test, y_prob_baseline):.3f})"
)

plt.plot(
    recall_only,
    precision_only,
    label=f"Engineered Features Only (PR-AUC = {average_precision_score(y_test, y_prob_only_eng):.3f})"
)

plt.plot(
    recall_eng,
    precision_eng,
    label=f"Original + Engineered (PR-AUC = {average_precision_score(y_test, y_prob_eng):.3f})"
)

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()
