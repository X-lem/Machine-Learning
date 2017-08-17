# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:, 1:2].values
Y = ds.iloc[:, 2].values


# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
lr = RandomForestRegressor(n_estimators = 300, random_state = 0)
lr.fit(X, Y)

# Predicting a new result
y_pred = lr.predict(6.5)


# Visualising the Random Forest Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lr.predict(X_grid), color = 'purple')
plt.title('Random Forest Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()