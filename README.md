\# 🚕 Taxi Demand Prediction



A machine learning project that predicts taxi demand based on \*\*time and pickup location\*\* using real-world NYC Yellow Taxi trip data.



The project processes millions of taxi trip records, performs exploratory data analysis and feature engineering, trains a Random Forest Regression model, evaluates its performance, and integrates the trained model into a Flask web application.



\---



\## 📌 Project Overview



Taxi demand changes depending on factors such as:



\- 🕐 Time of day

\- 📅 Day of the week

\- 📆 Day of the month

\- 📍 Pickup location



The goal of this project is to use these features to estimate the number of taxi trips expected for a particular hour and pickup location.



\### Project Workflow



```text

Raw Taxi Data

&#x20;     ↓

Data Inspection

&#x20;     ↓

Data Aggregation

&#x20;     ↓

Exploratory Data Analysis

&#x20;     ↓

Feature Engineering

&#x20;     ↓

Train / Test Split

&#x20;     ↓

Random Forest Regressor

&#x20;     ↓

Model Evaluation

&#x20;     ↓

Model Serialization

&#x20;     ↓

Flask Web Application

&#x20;     ↓

Taxi Demand Prediction

