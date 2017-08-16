# SVR

# Importing the dataset
ds = read.csv('Position_Salaries.csv')
ds = ds[2:3]


# Fitting the SVR Model to the dataset
# install.packages('e1071')
# library(e1071)
lr = svm(formula = Salary ~ ., 
         data = ds,
         type = 'eps-regression')

# Predicting a new result
y_pred = predict(lr, data.frame(Level = 6.5))

# Visualising the SVR Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             color = 'red') +
  geom_line(aes(x = ds$Level, y = predict(lr, newdata = ds)),
            color = 'purple') +
  ggtitle('SVR Model') +
  xlab('Level') +
  ylab('Salary')

# Visualising the SVR Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(ds$Level), max(ds$Level), 0.1)
ggplot() +
  geom_point(aes(x = ds$Level, y = ds$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid, y = predict(lr, newdata = data.frame(Level = x_grid))),
            color = 'purple') +
  ggtitle('SVR Model') +
  xlab('Level') +
  ylab('Salary')