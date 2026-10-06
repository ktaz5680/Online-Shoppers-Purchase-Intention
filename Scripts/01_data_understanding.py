import pandas as pd

# Load the dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

# Display basic dataset information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nRevenue Distribution:")
print(df["Revenue"].value_counts())

print("\nRevenue Percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)

print("\nSummary Statistics:")
print(df.describe())
