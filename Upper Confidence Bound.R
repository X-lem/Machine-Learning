# Upper Confidence Bound (UCB)

# Importing the dataset
ds = read.csv('Ads_CTR_Optimisation.csv')


# Implimenting (UCB)
N = 10000 # Number of rows in dataset
d = 10 # Number of ads
adSelected = integer()
NumberOfSelections = integer(d)
SumOfRewards = integer(d)
totalReward = 0

for (n in 1:N) {
  ad = 0
  maxUpperBound = 0
  for (i in 1:d) {
    if (NumberOfSelections[i] > 0) {
      averageReward = SumOfRewards[i] / NumberOfSelections[i]
      delta_i = sqrt(3 / 2 * log(n) / NumberOfSelections[i])
      upperBound = averageReward + delta_i
    } else {
      upperBound = 1e400
    }
    if (upperBound > maxUpperBound) {
      maxUpperBound = upperBound
      ad = i
    }
  }
  adSelected = append(adSelected, ad)
  NumberOfSelections[ad] = NumberOfSelections[ad] + 1
  reward = ds[n, ad]
  SumOfRewards[ad] = SumOfRewards[ad] + reward
  totalReward = totalReward + reward
}

# Visualising the results - Histogram
hist(adSelected, col = 'blue', main = 'Histogram of ads selections',
     xlab = 'Ads', ylab = 'Number of times each ad was selected')