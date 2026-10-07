import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Import data
eshop = pd.read_csv("data/online_shoppers_intention.csv")

# Separate predictors and target
X = eshop.drop("Revenue", axis=1).copy()
y = eshop["Revenue"].astype(int)

# Convert Weekend from Boolean to binary
X["Weekend"] = X["Weekend"].astype(int)

# Variables treated as categorical
categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType"
]

# One-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 80/20 stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Fit only on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Apply the same transformation to test data
X_test_encoded = preprocessor.transform(X_test)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))
