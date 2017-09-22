# Upper Confidence Bound (UCB)

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
ds = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implimenting the dataset
N = 10000 # Number of rows in dataset
d = 10 # Number of ads
adSelected = []
NumberOfSelections = [0] * d
SumOfRewards = [0] * d
totalReward = 0
for n in range(0, N):
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):
        if (NumberOfSelections[i] > 0):
            averageReward = SumOfRewards[i] / NumberOfSelections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / NumberOfSelections[i])
            upperBound = averageReward + delta_i
        else:
            upperBound = 1e400 # Ensures the first 10 rounds ads are selected sequentially
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i # Index
    adSelected.append(ad)
    NumberOfSelections[ad] = NumberOfSelections[ad] + 1
    reward = ds.values[n, ad]
    SumOfRewards[ad] = SumOfRewards[ad] + reward
    totalReward = totalReward + reward


# Visualising the results
plt.hist(adSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()