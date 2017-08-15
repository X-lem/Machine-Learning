# Regression Template

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Position_Salaries.csv")
X = ds.iloc[:, 1:2].values # Included the 2 in order to keep column 1 as a matrix
Y = ds.iloc[:, 2].values

# Splitting the dataset into a training set and test set
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""

# Feature scaling (if needed)
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting the Regression model to the Dataset


# Predicting a new result with Regression
Y_pred = regressor.predict(6.5)


# Visulizing the Regression results 
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'purple')
plt.title('Regression Model')
plt.xlabel('')
plt.ylabel('')
plt.show()


# Visulizing the Regression results (for better resulution)
X_Grid = np.arange(min(X), max(X), 0.1)
X_Grid = X_Grid.reshape((len(X_Grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'purple')
plt.title('Regression Model')
plt.xlabel('')
plt.ylabel('')
plt.show()