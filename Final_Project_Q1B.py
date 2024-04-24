import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


park_data = pd.read_csv("park.csv")

X = park_data.iloc[:, 1:]
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)
y = park_data.iloc[:, 0]   

model = LinearRegression()

kf = KFold(n_splits=10, shuffle=True, random_state=1)

scores = []

for train_index, test_index in kf.split(X_poly):
    X_poly_train, X_poly_test = X_poly[train_index], X_poly[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_poly_train, y_train)
    
    y_pred = model.predict(X_poly_test)
    
    score = mean_squared_error(y_test, y_pred)
    scores.append(score)

average_score = sum(scores) / len(scores)
print("Average MSE:", average_score)
