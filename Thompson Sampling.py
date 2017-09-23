# Thompson Sampling

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
ds = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implimenting Thompson Sampling
N = 10000 # Number of rows in dataset
d = 10 # Number of ads
adSelected = []
numberOfRewards_0 = [0] * d
numberOfRewards_1 = [0] * d
totalReward = 0
for n in range(0, N):
    ad = 0
    maxRandom = 0
    for i in range(0, d):
        randomBeta = random.betavariate(numberOfRewards_1[i] + 1, numberOfRewards_0[i] + 1)
        
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i # Index
    adSelected.append(ad)
    reward = ds.values[n, ad]
    if reward == 1:
        numberOfRewards_1[ad] = numberOfRewards_1[ad] + 1
    else:
        numberOfRewards_0[ad] = numberOfRewards_0[ad] + 1
    totalReward = totalReward + reward


# Visualising the results - Histogram
plt.hist(adSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()