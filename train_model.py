import pandas as pd
from sklearn.model_selection import train_test_split

# Load ML-ready data
df = pd.read_csv("data/ml_data.csv")

# Features
X = df[
    ["hour_of_day", "day_of_week", "day_of_month", "location"]
]

# Target
y = df["demand"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
import joblib

# Save the trained model
joblib.dump(model, "taxi_demand_model.pkl")

print("\nModel saved successfully!")