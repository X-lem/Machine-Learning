# Support Vector Machine (SVM)

# Importing the dataset
ds = read.csv('Social_Network_Ads.csv')
ds = ds[, 3:5]

# Encoding the target feature as factor
ds$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Purchased, SplitRatio = 0.75)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Feature scaling
trs[-3] = scale(trs[-3])
tes[-3] = scale(tes[-3])


# Fitting the classifier to the training set
# install.packages('e1071')
# library(e1071)
cl = svm(formula = Purchased ~ .,
         data = trs,
         type = 'C-classification',
         kernal = 'linear')


# Predicting the Test set Results
y_pred = predict(cl, newdata = tes[-3])

# Making the Confusion Matrix
cm = table(tes[, 3], y_pred)


# Visualising the Training set results
install.packages('ElemStatLearn')
library(ElemStatLearn)
set = trs
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(cl, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Support Vector Machine (SVM)(Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'cyan', 'purple'))


# Visualising the Test set results
library(ElemStatLearn)
set = tes
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(cl, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Support Vector Machine (SVM) (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'cyan', 'purple'))