# Data Preprocessing

# Importing the dataset
ds = read.csv('Salary_Data.csv')

# ds = ds[, 2:3] # If needing to select specific indexes


# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Salary, SplitRatio = 2/3)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Fitting Simple Linear Regression to the training set
lr = lm(formula = Salary ~ YearsExperience, data = trs)

# Predicting the test set restults
y_pred = predict(lr, newdata = tes)

# Visualising the Training set results
# install.packages('ggplot2')
# library(ggplot2)
ggplot() +
  geom_point(aes(x = trs$YearsExperience, y = trs$Salary), colour = 'red') +
  geom_line(aes(x = trs$YearsExperience, y = predict(lr, newdata = trs)), colour = 'purple') +
  ggtitle('Salary vs Experience') +
  xlab('Years of Experience') +
  ylab('Salary')
 
# Visualising the Test set results
ggplot() +
  geom_point(aes(x = tes$YearsExperience, y = tes$Salary), colour = 'red') +
  geom_line(aes(x = trs$YearsExperience, y = predict(lr, newdata = trs)), colour = 'purple') +
  ggtitle('Salary vs Experience') +
  xlab('Years of Experience') +
  ylab('Salary')


  