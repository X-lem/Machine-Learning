# Regression Template

# Importing the dataset
ds = read.csv('Position_Salaries.csv')
ds = ds[2:3]

# Fitting the Regression Model to the dataset
# install.packages('randomForest')
# library(randomForest)
set.seed(1234)
lr = randomForest(x = ds[1], 
                  y = ds$Salary,
                  ntree = 500)

# Predicting a new result
y_pred = predict(lr, data.frame(Level = 6.5))


# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(ds$Level), max(ds$Level), 0.01)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid, y = predict(lr, newdata = data.frame(Level = x_grid))),
            color = 'purple') +
  ggtitle('Regression Model') +
  xlab('Level') +
  ylab('Salary')