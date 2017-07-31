# Machine Learning Template

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Data.csv")
X = ds.iloc[:, : -1].values # [Rows, Columns] : = take all
Y = ds.iloc[:, 3].values

# Splitting the dataset into a training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling (if needed)
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""