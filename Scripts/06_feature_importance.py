import runpy
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Run the modeling script and collect its variables
model = runpy.run_path("Scripts/04_modeling.py")

preprocessor_eng = model["preprocessor_eng"]
X_train_eng = model["X_train_eng"]
X_test_eng = model["X_test_eng"]
y_train = model["y_train"]
y_test = model["y_test"]

# Build the engineered Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_eng),
    ("model", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    ))
])

# Fit the model
rf_pipeline.fit(X_train_eng, y_train)

# Permutation importance using PR-AUC
perm = permutation_importance(
    rf_pipeline,
    X_test_eng,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=42
)

# Create importance table
importance = pd.DataFrame({
    "Feature": X_test_eng.columns,
    "Importance": perm.importances_mean
}).sort_values(
    "Importance",
    ascending=False
)

print("\nPermutation Feature Importance")
print(importance.round(4))
