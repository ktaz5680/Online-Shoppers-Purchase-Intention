import os
import runpy

# Make sure output folders exist
os.makedirs("Outputs/Results", exist_ok=True)
os.makedirs("Outputs/Figures", exist_ok=True)

# Run evaluation
evaluation = runpy.run_path("Scripts/05_evaluation.py")

# Save model performance results
results = evaluation["results"]

results.to_csv(
    "Outputs/Results/model_performance.csv",
    index=False
)

# Run feature importance
feature_results = runpy.run_path(
    "Scripts/06_feature_importance.py"
)

# Save feature importance results
importance = feature_results["importance"]

importance.to_csv(
    "Outputs/Results/feature_importance.csv",
    index=False
)

print("Project results saved successfully.")
print("See the Outputs/Results folder.")
