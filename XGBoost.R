# XGBoost

# Importing the dataset
ds = read.csv('Churn_Modelling.csv')
ds = ds[4:14]

# Encoding categorical data
ds$Geography = as.numeric(factor(ds$Geography,
                                 levels = c('France', 'Spain', 'Germany'),
                                 labels = c(1, 2, 3)))
ds$Gender = as.numeric(factor(ds$Gender,
                              levels = c('Female', 'Male'),
                              labels = c(1, 2)))

# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Exited, SplitRatio = 0.8)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Fitting XGboost to the training set
# install.packages('xgboost')
library(xgboost)

cl = xgboost(data = as.matrix(trs[-11]), label = trs$Exited, nrounds = 10)

# Applying K-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(trs$Exited, k = 10)
cv = lapply(folds, function(x) {
  trsFold = trs[-x, ]
  tesFold = trs[x, ]
  cl = xgboost(data = as.matrix(trs[-11]), label = trs$Exited, nrounds = 10)
  y_pred = predict(cl, newdata = as.matrix(tesFold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(tesFold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
