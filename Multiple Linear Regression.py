# Multiple Linear Regression (MLR)

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("50_Startups.csv")
X = ds.iloc[:, : -1].values # [Rows, Columns] : = take all
Y = ds.iloc[:, 4].values

# Encoding catagorical data
# Dependant variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])
# Create dummy variables
oe = OneHotEncoder(categorical_features = [3])
X = oe.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:] # Remove the first column - X[0]


# Splitting the dataset into a training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting the MLR to the training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # Adding a column of 1's
# Optimal matrix of features (Significant level is 5%)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
lr_ols = sm.OLS(endog = Y, exog = X_opt).fit()
lr_ols.summary()

# Remove [2]
X_opt = X[:, [0, 1, 3, 4, 5]]
lr_ols = sm.OLS(endog = Y, exog = X_opt).fit()
lr_ols.summary()

# Remove [1]
X_opt = X[:, [0, 3, 4, 5]]
lr_ols = sm.OLS(endog = Y, exog = X_opt).fit()
lr_ols.summary()

# Remove [4]
X_opt = X[:, [0, 3, 5]]
lr_ols = sm.OLS(endog = Y, exog = X_opt).fit()
lr_ols.summary()

# Remove [5]
X_opt = X[:, [0, 3]]
lr_ols = sm.OLS(endog = Y, exog = X_opt).fit()
lr_ols.summary()
