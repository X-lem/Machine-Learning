# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:, 1:2].values
Y = ds.iloc[:, 2].values


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.tree import DecisionTreeRegressor
lr = DecisionTreeRegressor(random_state = 0)
lr.fit(X, Y)


# Predicting a new result
y_pred = lr.predict(6.5)

# Visualising the Decision Tree Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lr.predict(X_grid), color = 'purple')
plt.title('Decision Tree Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()