from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("taxi_demand_model.pkl")


# =========================================================
# LOAD ML DATA
# =========================================================

df = pd.read_csv("data/ml_data.csv")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

model_r2 = 0.9605624403143134
model_accuracy = round(model_r2 * 100, 1)


# =========================================================
# HOURLY DEMAND
# =========================================================

hourly_demand = (
    df.groupby("hour_of_day")["demand"]
    .sum()
    .reindex(range(24), fill_value=0)
)

hourly_data = list(
    zip(
        hourly_demand.index.tolist(),
        hourly_demand.values.tolist()
    )
)

hourly_max = (
    max(hourly_demand.values)
    if len(hourly_demand) > 0
    else 1
)


# =========================================================
# TOP PICKUP LOCATIONS
# =========================================================

location_demand = (
    df.groupby("location")["demand"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_location_data = list(
    zip(
        location_demand.index.astype(str).tolist(),
        location_demand.values.tolist()
    )
)

location_max = (
    max(location_demand.values)
    if len(location_demand) > 0
    else 1
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

feature_names = [
    "Pickup Location",
    "Hour of Day",
    "Day of Week",
    "Day of Month"
]

# The model was trained in this original feature order
original_feature_names = [
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "location"
]

importance_values = model.feature_importances_

feature_importance = []

for name, value in zip(
    original_feature_names,
    importance_values
):

    display_name = {
        "hour_of_day": "Hour of Day",
        "day_of_week": "Day of Week",
        "day_of_month": "Day of Month",
        "location": "Pickup Location"
    }[name]

    feature_importance.append({
        "name": display_name,
        "value": round(value * 100, 2)
    })


# Sort from highest to lowest
feature_importance.sort(
    key=lambda x: x["value"],
    reverse=True
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    demand_level = None
    insight = None

    selected_hour = None
    selected_day = None
    selected_date = None
    selected_location = None


    # =====================================================
    # PREDICTION
    # =====================================================

    if request.method == "POST":

        selected_hour = int(
            request.form["hour"]
        )

        selected_day = int(
            request.form["day"]
        )

        selected_date = int(
            request.form["date"]
        )

        selected_location = int(
            request.form["location"]
        )


        # -------------------------------------------------
        # Create input for the ML model
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "hour_of_day": [selected_hour],

            "day_of_week": [selected_day],

            "day_of_month": [selected_date],

            "location": [selected_location]

        })


        # -------------------------------------------------
        # Predict taxi demand
        # -------------------------------------------------

        prediction = round(
            model.predict(input_data)[0]
        )


        # -------------------------------------------------
        # Determine demand level
        # -------------------------------------------------

        if prediction < 100:

            demand_level = "Low"

        elif prediction < 300:

            demand_level = "Moderate"

        elif prediction < 500:

            demand_level = "High"

        else:

            demand_level = "Very High"


        # -------------------------------------------------
        # Smart insight
        # -------------------------------------------------

        if demand_level == "Very High":

            insight = (
                "Demand is expected to be very high. "
                "More taxi availability may be useful during this period."
            )

        elif demand_level == "High":

            insight = (
                "Demand is expected to be high. "
                "This could be a busy period for taxi services."
            )

        elif demand_level == "Moderate":

            insight = (
                "Demand is expected to be moderate. "
                "Normal taxi availability should be sufficient."
            )

        else:

            insight = (
                "Demand is expected to be relatively low. "
                "Taxi availability may not need to be increased significantly."
            )


    # =====================================================
    # SEND DATA TO HTML
    # =====================================================

    return render_template(

        "index.html",

        prediction=prediction,

        demand_level=demand_level,

        insight=insight,

        selected_hour=selected_hour,

        selected_day=selected_day,

        selected_date=selected_date,

        selected_location=selected_location,

        model_accuracy=model_accuracy,

        hourly_data=hourly_data,

        hourly_max=hourly_max,

        top_location_data=top_location_data,

        location_max=location_max,

        feature_importance=feature_importance
    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)