import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Load the data
file_path = "combined_metrics_and_stock_data.parquet"
df = pd.read_parquet(file_path)

# Features and target
features = [
    "Volume", "Histogram", "Signal", "MACD", "credibility_value",
    "outlook_value", "referential_depth_value", "sentiment_value",
    "z_score_scaled", "max_words_normalized"
]
target = "close_pct_change"

# Shift the target column by -1 to predict the next day's change
df["next_close_pct_change"] = df[target].shift(-1)

# Drop the last row since it won't have a valid target
df = df.dropna()

# Split features and target
X = df[features]
y = df["next_close_pct_change"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix, optimized for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    "objective": "reg:squarederror",  # Regression task
    "max_depth": 5,                  # Depth of trees
    "eta": 0.1,                      # Learning rate
    "n_estimators": 100              # Number of trees
}

# Train the model
num_boost_round = 100
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

# Predict
y_pred = model.predict(dtest)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Show feature importance
xgb.plot_importance(model)

# TESTING Evaluation
# Create a DataFrame for visualization
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

# Add an index column for test data points
results_df["Index"] = range(len(y_test))

# Melt the DataFrame for easier plotting with Plotly Express
results_df_melted = results_df.melt(id_vars=["Index"], value_vars=["Actual", "Predicted"],
                                    var_name="Type", value_name="Next Close Percentage Change")

# Plot actual vs predicted using Plotly Express
fig_performance = px.line(
    results_df_melted,
    x="Index",
    y="Next Close Percentage Change",
    color="Type",
    title="Actual vs Predicted Next Day Close Percentage Change",
    labels={"Index": "Test Data Points", "Next Close Percentage Change": "Percentage Change"}
)

# Feature Importance
# Get feature importance from the trained XGBoost model
importance = model.get_score(importance_type='weight')

# Convert feature importance to a DataFrame
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])

# Sort by importance for better visualization
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance using Plotly Express
fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation='h',
    title="Feature Importance",
    labels={"Importance": "Importance Score", "Feature": "Features"},
    height=400
)

# Show the plots
fig_performance.show()
fig_importance.show()

# Save model
model.save_model("xgboost_pcib_model.json")