# Artificial Neural Network (ANN)

# Installing Theano
# conda install theano pygpu
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


##### Data Preprocessing

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

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


##### Creating the ANN

# Import the Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
cl = Sequential()

# Create the input layer and the first hidden layer
cl.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
cl.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) # Second hidden layer
cl.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # Output layer

# Compiling the ANN
cl.compile(optimizer = 'adam', loss = 'binary_crossentropy', metics = ['accuracy'])


##### Making the predictions



# Predicint the test set results
y_pred = cl.predict(X_test)

# Making the confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)