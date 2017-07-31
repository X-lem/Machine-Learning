# Data Preprocessing

# Importing the dataset
ds = read.csv('Data.csv')

# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Purchased, SplitRatio = 0.8)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Feature scaling
trs[, 2:3] = scale(trs[2:3])
tes[2:3] = scale(tes[2:3])