# Natural Language Processing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3) # Quoting is for ignoring double quotes

# Cleaning the text
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', ds['Review'][i]) # Keep only letters
    review = review.lower() # To lowercase
    review = review.split() # Split into an array of individual words
    # Remove usless words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review,) # Join the words back together into a string
    corpus.append(review)
    
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Taking the top 1500 words
X = cv.fit_transform(corpus).toarray()
Y = ds.iloc[:, 1].values

####### Naive Bayes

# Splitting the dataset into a training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB
cl = GaussianNB()
cl.fit(X_train, Y_train)

# Predicint the test set results
y_pred = cl.predict(X_test)

# Making the confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


####### End Naive Bayes

####### Decision Tree Classification

# Splitting the dataset into a training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# Fitting Decision Tree Classification to the Training Set
from sklearn.tree import DecisionTreeClassifier
cl = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
cl.fit(X_train, Y_train)

# Predicint the test set results
y_pred = cl.predict(X_test)

# Making the confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

####### End Decision Tree Classification

####### Random Forest Classification
# Splitting the dataset into a training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# Fitting the Random Forest classifier to the Training Set
from sklearn.ensemble import RandomForestClassifier
cl = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
cl.fit(X_train, Y_train)

# Predicint the test set results
y_pred = cl.predict(X_test)

# Making the confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

####### End Random Forest Classification