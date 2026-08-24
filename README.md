# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required Python libraries: Pandas, NumPy, Scikit-learn.
2. Load the weather dataset using `pd.read_csv()`.
3. Display the first five rows of the dataset.
4. Convert the `time` column into datetime format.
5. Extract `month`, `day`, and `hour` from the time column.
6. Identify all numerical columns and replace missing values with their respective mean values.
7. Select the input features:

   * Humidity (`hum`)
   * Pressure (`pressure`)
   * Wind speed (`wind_speed`)
   * Month
   * Day
   * Hour
8. Select the target variables:

   * Temperature (`tem`)
   * PM2.5 (`pm2_5`)
   * Total Solar Radiation (`tsr`)
9. Split the dataset into **80% training data and 20% testing data**.
10. Create a **Random Forest Regressor** with 100 decision trees.
11. Train the model using the training data.
12. Predict the target values using the testing data.
13. Evaluate the model using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

14. Perform **5-fold cross-validation** using the R² scoring method.
15. Calculate and display the average cross-validation score.
16. End the program.



## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('weather-station-eee-block_2024_07_13.csv')

# Display first five rows
print("First Five Rows of Dataset")
print(df.head())

# Convert time column into datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract date and time features
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Handle missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Select input features
X = df[['hum', 'pressure', 'wind_speed', 'month', 'day', 'hour']]

# Select target variables
Y = df[['tem', 'pm2_5', 'tsr']]

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, Y_train)

# Predict output values
Y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Display evaluation metrics
print("\nModel Evaluation Metrics")
print("R2 Score :", r2)
print("MAE :", mae)
print("RMSE :", rmse)

# Perform cross-validation
cv_scores = cross_val_score(
    model,
    X,
    Y,
    cv=5,
    scoring='r2'
)

# Display cross-validation results
print("\nCross Validation Scores")
print(cv_scores)

print("\nAverage Cross Validation Score")
print(cv_scores.mean())
```

## Output:

<img width="821" height="696" alt="Screenshot 2026-08-24 125610" src="https://github.com/user-attachments/assets/4936d368-f0e3-4124-aba1-2603eb21b666" />

## Result:

The Random Forest Regression model was successfully implemented for predicting **temperature, PM2.5, and total solar radiation** from the weather dataset.

The program displays:

* First five rows of the dataset
* **R² Score**
* **MAE**
* **RMSE**
* **5 Cross-Validation Scores**
* **Average Cross-Validation Score**

