# Thompson Sampling

# Importing the dataset
ds = read.csv('Ads_CTR_Optimisation.csv')


# Implimenting Thompson Sampling
N = 10000 # Number of rows in dataset
d = 10 # Number of ads
adSelected = integer()
numberOfRewards_0 = integer(d)
numberOfRewards_1 = integer(d)
totalReward = 0

for (n in 1:N) {
  ad = 0
  maxRandom = 0
  for (i in 1:d) {
    randomBeta = rbeta(n = 1,
                       shape1 = numberOfRewards_1[i] + 1,
                       shape2 = numberOfRewards_0[i] + 1)
    if (randomBeta > maxRandom) {
      maxRandom = randomBeta
      ad = i
    }
  }
  
  adSelected = append(adSelected, ad)
  reward = ds[n, ad]
  if (reward == 1) {
    numberOfRewards_1[ad] = numberOfRewards_1[ad] + 1
  } else {
    numberOfRewards_0[ad] = numberOfRewards_0[ad] + 1
  }
  totalReward = totalReward + reward
}

# Visualising the results - Histogram
hist(adSelected, col = 'blue', main = 'Histogram of ads selections',
     xlab = 'Ads', ylab = 'Number of times each ad was selected')