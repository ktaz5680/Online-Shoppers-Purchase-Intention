import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Import data
eshop = pd.read_csv("data/online_shoppers_intention.csv")

# --------------------------------------------------
# Original features
# --------------------------------------------------

X = eshop.drop("Revenue", axis=1).copy()
y = eshop["Revenue"].astype(int)

X["Weekend"] = X["Weekend"].astype(int)

categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType"
]

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

# Same 80/20 stratified split used in the notebook
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Model 1: Original Features
rf_baseline = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

rf_baseline.fit(X_train_encoded, y_train)

y_pred_baseline = rf_baseline.predict(X_test_encoded)
y_prob_baseline = rf_baseline.predict_proba(X_test_encoded)[:, 1]


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

eshop["TotalPages"] = (
    eshop["Administrative"]
    + eshop["Informational"]
    + eshop["ProductRelated"]
)

eshop["TotalDuration"] = (
    eshop["Administrative_Duration"]
    + eshop["Informational_Duration"]
    + eshop["ProductRelated_Duration"]
)

eshop["AverageTimePerPage"] = (
    eshop["TotalDuration"] / eshop["TotalPages"]
)

eshop["ProductPageShare"] = (
    eshop["ProductRelated"] / eshop["TotalPages"]
)

eshop["AverageTimePerPage"] = (
    eshop["AverageTimePerPage"].fillna(0)
)

eshop["ProductPageShare"] = (
    eshop["ProductPageShare"].fillna(0)
)


# --------------------------------------------------
# Model 2: Engineered Features Only
# --------------------------------------------------

engineered_features = [
    "TotalPages",
    "TotalDuration",
    "AverageTimePerPage",
    "ProductPageShare"
]

# Use the same training and testing observations
X_train_only_eng = eshop.loc[
    X_train.index,
    engineered_features
]

X_test_only_eng = eshop.loc[
    X_test.index,
    engineered_features
]

rf_only_eng = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

rf_only_eng.fit(X_train_only_eng, y_train)

y_pred_only_eng = rf_only_eng.predict(X_test_only_eng)
y_prob_only_eng = rf_only_eng.predict_proba(
    X_test_only_eng
)[:, 1]


# --------------------------------------------------
# Model 3: Original + Engineered Features
# --------------------------------------------------

X_eng = eshop.drop("Revenue", axis=1).copy()
X_eng["Weekend"] = X_eng["Weekend"].astype(int)

# Use the same observations as the original model
X_train_eng = X_eng.loc[X_train.index]
X_test_eng = X_eng.loc[X_test.index]

preprocessor_eng = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_train_eng_encoded = preprocessor_eng.fit_transform(
    X_train_eng
)

X_test_eng_encoded = preprocessor_eng.transform(
    X_test_eng
)

rf_engineered = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

rf_engineered.fit(
    X_train_eng_encoded,
    y_train
)

y_pred_eng = rf_engineered.predict(
    X_test_eng_encoded
)

y_prob_eng = rf_engineered.predict_proba(
    X_test_eng_encoded
)[:, 1]

print("All three Random Forest models trained successfully.")
