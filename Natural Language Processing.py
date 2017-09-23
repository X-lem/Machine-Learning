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
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
