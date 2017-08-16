# Regression Template

# Importing the dataset
ds = read.csv('Position_Salaries.csv')
ds = ds[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(ds$Salary, SplitRatio = 2/3)
# trs = subset(ds, split == TRUE) # Training Set
# tes = subset(ds, split == FALSE) # Test Set

# Feature Scaling
# trs = scale(trs)
# tes = scale(tes)

# Fitting the Regression Model to the dataset
# Create your regressor here
#lr = 

# Predicting a new result
y_pred = predict(lr, data.frame(Level = 6.5))

# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             color = 'red') +
  geom_line(aes(x = ds$Level, y = predict(lr, newdata = ds)),
            color = 'purple') +
  ggtitle('Regression Model') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(ds$Level), max(ds$Level), 0.1)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid, y = predict(lr, newdata = data.frame(Level = x_grid))),
            color = 'purple') +
  ggtitle('Regression Model') +
  xlab('Level') +
  ylab('Salary')