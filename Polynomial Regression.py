# Polynomial Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Position_Salaries.csv")
X = ds.iloc[:, 1:2].values # Included the 2 in order to keep column 1 as a matrix
Y = ds.iloc[:, 2].values


# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, Y)

# Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4)
X_Poly = pr.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(X_Poly, Y)

# Visulizing the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lr.predict(X), color = 'purple')
plt.title('Employee Salery')
plt.xlabel('Position')
plt.ylabel('salery')
plt.show()

# Visulizing the Polynomial Regression results
X_Grid = np.arange(min(X), max(X), 0.1)
X_Grid = X_Grid.reshape((len(X_Grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_Grid, lr2.predict(pr.fit_transform(X_Grid)), color = 'purple')
plt.title('Employee Salery')
plt.xlabel('Position')
plt.ylabel('salery')
plt.show()


# Predicting a new result with Linear Regression
lr.predict(6.5)

# Predicting a new result with Polynomial Regression
lr2.predict(pr.fit_transform(6.5))