import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your data
park_data = pd.read_csv("park.csv")

# Assuming the first column is your target variable
X = park_data.iloc[:, 1:]  # features
y = park_data.iloc[:, 0]   # target variable

# Initialize the model
model = LinearRegression()

# Setup 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# To store the results
scores = []

# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the error (e.g., MSE)
    score = mean_squared_error(y_test, y_pred)
    scores.append(score)

# Output the average error across the folds
average_score = sum(scores) / len(scores)
print("Average MSE:", average_score)
