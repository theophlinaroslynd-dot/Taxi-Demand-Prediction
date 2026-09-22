import pandas as pd

# Load our processed demand data
df = pd.read_csv("data/taxi_demand.csv")

# Convert hour column to datetime
df["hour"] = pd.to_datetime(df["hour"])

# Create useful time features
df["hour_of_day"] = df["hour"].dt.hour
df["day_of_week"] = df["hour"].dt.dayofweek
df["day_of_month"] = df["hour"].dt.day

# Rename location column
df["location"] = df["PULocationID"]

# Keep only the features we need
df = df[
    ["hour_of_day", "day_of_week", "day_of_month", "location", "demand"]
]

# Save the ML-ready dataset
df.to_csv("data/ml_data.csv", index=False)

print("Feature engineering completed!")
print("\nFirst 5 rows:")
print(df.head())

print("\nML Dataset Shape:")
print(df.shape)