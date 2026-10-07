import pandas as pd

# Import data
eshop = pd.read_csv("data/online_shoppers_intention.csv")

# Feature Engineering

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

# Replace undefined values caused by zero pages
eshop["AverageTimePerPage"] = eshop["AverageTimePerPage"].fillna(0)
eshop["ProductPageShare"] = eshop["ProductPageShare"].fillna(0)

print(
    eshop[
        [
            "TotalPages",
            "TotalDuration",
            "AverageTimePerPage",
            "ProductPageShare"
        ]
    ].head()
)
