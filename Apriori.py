# Apriori

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
trans = []
for i in range(0, 7501):
    trans.append([str(ds.values[i, j]) for j in range(0, 20)])
    

# Training Aproiri on the Dataset
from apyori import apriori # Make sure 'apyori.py' is in your file folder
rules = apriori(trans, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the reults
results = list(rules)