import pandas as pd
import matplotlib.pyplot as plt

# Load processed taxi demand data
df = pd.read_csv("data/taxi_demand.csv")

# Convert hour column to datetime
df["hour"] = pd.to_datetime(df["hour"])

# Calculate total demand for each hour of the day
hourly_demand = df.groupby(
    df["hour"].dt.hour
)["demand"].sum()

# Create the graph
plt.figure(figsize=(10, 5))
plt.plot(hourly_demand.index, hourly_demand.values)

plt.xlabel("Hour of the Day")
plt.ylabel("Number of Taxi Trips")
plt.title("Taxi Demand by Hour")

plt.xticks(range(24))
plt.grid()

plt.show()
# Create a day of week column
df["day_of_week"] = df["hour"].dt.day_name()

# Calculate total demand for each day
daily_demand = df.groupby("day_of_week")["demand"].mean()

# Arrange days in the correct order
days = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

daily_demand = daily_demand.reindex(days)

# Create the graph
plt.figure(figsize=(10, 5))
plt.bar(daily_demand.index, daily_demand.values)

plt.xlabel("Day of the Week")
plt.ylabel("Number of Taxi Trips")
plt.title("Taxi Demand by Day of the Week")

plt.xticks(rotation=45)
plt.grid(axis="y")

plt.show()
# Calculate total demand for each pickup location
location_demand = df.groupby("PULocationID")["demand"].sum()

# Select the top 10 locations
top_locations = location_demand.sort_values(ascending=False).head(10)

# Create the graph
plt.figure(figsize=(10, 5))
plt.bar(top_locations.index.astype(str), top_locations.values)

plt.xlabel("Pickup Location ID")
plt.ylabel("Number of Taxi Trips")
plt.title("Top 10 Pickup Locations by Taxi Demand")

plt.grid(axis="y")

plt.show()