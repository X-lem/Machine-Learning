# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#
# Make sure xgboost in the local folder this python script is in.

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Churn_Modelling.csv")
X = ds.iloc[:, 3:13].values # [Rows, Columns] : = take all
Y = ds.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Remove the first dummy variable


# Splitting the dataset into a training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting XGboost to the training set
from xgboost import XGBClassifier
cl = XGBClassifier()
cl.fit(X_train, Y_train)

# Predicint the test set results
y_pred = cl.predict(X_test)
y_pred = (y_pred > 0.5) # True if larger than 0.5

# Making the confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cl, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()