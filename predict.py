import joblib
import pandas as pd

# Load the trained model
model = joblib.load("taxi_demand_model.pkl")

# Get input from the user
hour = int(input("Enter hour (0-23): "))
day = int(input("Enter day of week (0=Monday, 6=Sunday): "))
date = int(input("Enter day of month (1-31): "))
location = int(input("Enter pickup location ID: "))

# Create input data
input_data = pd.DataFrame({
    "hour_of_day": [hour],
    "day_of_week": [day],
    "day_of_month": [date],
    "location": [location]
})

# Make prediction
prediction = model.predict(input_data)

print("\nPredicted Taxi Demand:", round(prediction[0]), "trips")