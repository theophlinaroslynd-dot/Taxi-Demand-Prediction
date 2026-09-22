import pandas as pd

# Load only the columns we need
df = pd.read_parquet(
    "data/yellow_tripdata_2025-01.parquet",
    columns=["tpep_pickup_datetime", "PULocationID"]
)

# Create an hour column
df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")

print(df.head())

# Count taxi trips for each hour and location
demand = df.groupby(
    ["hour", "PULocationID"]
).size().reset_index(name="demand")

print("\nTaxi Demand:")
print(demand.head(20))

# Save the processed demand dataset
demand.to_csv("data/taxi_demand.csv", index=False)

print("\nProcessed dataset saved successfully!")