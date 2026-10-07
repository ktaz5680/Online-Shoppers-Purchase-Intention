import pandas as pd

# Import data
eshop = pd.read_csv("data/online_shoppers_intention.csv")

# Basic dataset information
print("Dataset shape:", eshop.shape)

print("\nColumn names:")
print(eshop.columns.tolist())

print("\nData types:")
print(eshop.dtypes)

print("\nFirst five rows:")
print(eshop.head())

# Missing values
missing_values = eshop.isnull().sum()

print("\nMissing values by variable:")
print(missing_values)

print("\nTotal missing values:", missing_values.sum())

# Identical / duplicate rows
duplicate_count = eshop.duplicated().sum()

print("\nNumber of identical rows:", duplicate_count)
print(
    "Percentage of dataset:",
    round((duplicate_count / len(eshop)) * 100, 2),
    "%"
)

# Revenue target distribution
revenue_counts = eshop["Revenue"].value_counts()
revenue_percent = eshop["Revenue"].value_counts(normalize=True) * 100

target_summary = pd.DataFrame({
    "Count": revenue_counts,
    "Percentage": revenue_percent
})

print("\nRevenue target distribution:")
print(target_summary)
