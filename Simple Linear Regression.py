# Simple Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Salary_Data.csv")
X = ds.iloc[:, :-1].values # [Rows, Columns] : = take all
Y = ds.iloc[:, 1].values

# Splitting the dataset into a training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Predict the test set results
y_pred = lr.predict(X_test)

# Visulizing the training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'purple')
plt.title('Salery vs. Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salery')
plt.show()


# Visulizing the test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'purple')
plt.title('Salery vs. Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salery')
plt.show()