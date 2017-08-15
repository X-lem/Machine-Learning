# Polynomial Regression

# Importing the dataset
ds = read.csv('Position_Salaries.csv')

ds = ds[2:3] # Selecting only Level and Salary


# Splitting the dataset into a training set and test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(ds$Purchased, SplitRatio = 0.8)
# trs = subset(ds, split == TRUE) # Training Set
# tes = subset(ds, split == FALSE) # Test Set

# Feature scaling (if needed)
# trs[, 2:3] = scale(trs[2:3])
# tes[2:3] = scale(tes[2:3])

# Fitting Linear Regression to the dataset
lr = lm(formula = Salary ~ ., data = ds)

# Fitting Polynomial Regression to the dataset
ds$Level2 = ds$Level^2
ds$Level3 = ds$Level^3
ds$Level4 = ds$Level^4
pr = lm(formula = Salary ~ ., data = ds)


# Visulizing the Linear Regression results
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
            color = 'red') +
  geom_line(aes(x = ds$Level, y = predict(lr, newdata = ds)), 
            color = 'purple') +
  ggtitle('Linear Regression') +
  xlab('Level') +
  ylab('Salary')


# Visulizing the Polynomial Regression results
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             color = 'red') +
  geom_line(aes(x = ds$Level, y = predict(pr, newdata = ds)), 
            color = 'purple') +
  ggtitle('Polynomial Regression') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
Y_pred = predict(lr, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
Y_pred = predict(pr, data.frame(Level = 6.5,
                                Level2 = 6.5^2,
                                Level3 = 6.5^3,
                                Level4 = 6.5^4))

