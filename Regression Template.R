# Regression Template

# Importing the dataset
ds = read.csv('Position_Salaries.csv')
ds = ds[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(ds$Salary, SplitRatio = 2/3)
# training_set = subset(ds, split == TRUE)
# test_set = subset(ds, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             colour = 'red') +
  geom_line(aes(x = ds$Level, y = predict(regressor, newdata = ds)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(ds$Level), max(ds$Level), 0.1)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')