import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import Polynomial

# Step 1: Generate Noisy Sinusoidal Data
np.random.seed(42)  # For reproducibility

# Generate x values
x = np.linspace(0, 2 * np.pi, 500)  # 500 points between 0 and 2π

# Generate true sinusoidal signal
y_true = np.sin(x)

# Add noise
noise = np.random.normal(0, 0.2, size=x.shape)  # Gaussian noise
y_noisy = y_true + noise

# Reshape for XGBoost input
X = x.reshape(-1, 1)  # Features (1D array to 2D array)
y = y_noisy           # Target

# Step 2: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and Train XGBoost Model
# Convert to DMatrix, the optimized data format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    "objective": "reg:squarederror",  # Regression task
    "max_depth": 3,                  # Depth of trees
    "eta": 0.1,                      # Learning rate
    "n_estimators": 100              # Number of trees
}

# Train the model
num_boost_round = 100
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

# Step 4: Make Predictions
y_pred = model.predict(dtest)

# Function to fit a polynomial to predictions
def fit_predictions(X_test, y_pred, degree=5):
    """
    Fit a polynomial to predictions and return the fitted function.

    Parameters:
        X_test (np.ndarray): Test input data.
        y_pred (np.ndarray): Predicted values from the model.
        degree (int): Degree of the polynomial to fit.

    Returns:
        Polynomial: Fitted polynomial function.
    """
    poly = Polynomial.fit(X_test.flatten(), y_pred, degree)
    return poly

# Fit a polynomial to the predictions
fitted_poly = fit_predictions(X_test, y_pred, degree=5)

# Generate smooth values using the fitted function
x_smooth = np.linspace(0, 2 * np.pi, 1000)
y_fitted = fitted_poly(x_smooth)

# Step 5: Evaluate and Visualize
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy, label="Noisy Data", color="gray", alpha=0.5, s=10)
plt.plot(x, y_true, label="True Signal", color="green", linewidth=2)
plt.scatter(X_test, y_pred, label="Predictions", color="red", alpha=0.8, s=20)
plt.plot(x_smooth, y_fitted, label="Fitted Function", color="blue", linewidth=2, linestyle="--")
plt.legend()
plt.title("XGBoost Regression and Fitted Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
