import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load the Auto dataset (adjust the file path if necessary)
auto_data = pd.read_csv('Auto.csv')

# Ensure that horsepower column is numeric (since it can sometimes be imported as a string due to missing values)
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')

# Drop rows with missing values (if any)
auto_data.dropna(subset=['horsepower', 'mpg'], inplace=True)

# Define the response (mpg) and predictor (horsepower)
X = auto_data['horsepower']
y = auto_data['mpg']

# Add a constant term to the predictor to account for the intercept
X = sm.add_constant(X)

# Perform the simple linear regression
model = sm.OLS(y, X).fit()

# Summarize the model
print(model.summary())

# (a) Prediction for horsepower = 98
horsepower_98 = np.array([1, 98])  # [1, horsepower_value] to account for the intercept
prediction = model.get_prediction(horsepower_98)

# Print predicted value, confidence interval, and prediction interval
prediction_summary = prediction.summary_frame(alpha=0.05)
print("\nPrediction for horsepower = 98:")
print(prediction_summary[['mean', 'mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper']])

# (b) Scatter plot of the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(auto_data['horsepower'], auto_data['mpg'], label='Data', color='blue')

# Plot the regression line
plt.plot(auto_data['horsepower'], model.fittedvalues, label='Regression Line', color='red')

# Labels and title
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('MPG vs Horsepower')
plt.legend()
plt.show()

# (c) Diagnostic Plots
# Residuals vs Fitted Plot
plt.figure(figsize=(10, 6))

# Residuals plot
plt.subplot(1, 2, 1)
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# Q-Q plot
plt.subplot(1, 2, 2)
sm.qqplot(model.resid, line='s', ax=plt.gca())
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()