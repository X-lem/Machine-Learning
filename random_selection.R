# Random Selection

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
N = 10000
d = 10
adDisplayed = integer(0)
totalReward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  adDisplayed = append(adDisplayed, ad)
  reward = dataset[n, ad]
  totalReward = totalReward + reward
}

# Visualising the results
hist(adDisplayed,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was displayed')