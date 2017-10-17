# Principal Component Analysis (PCA)

# Importing the dataset
ds = read.csv('Wine.csv')

# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Customer_Segment, SplitRatio = 0.8)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Feature scaling (if needed)
trs[-14] = scale(trs[-14])
tes[-14] = scale(tes[-14])


# Applying PCA
# install.packages('caret')
# install.packages('PreProcess')
library(caret)
# install.packages('e1071')
library(e1071)

pca = preProcess(x = trs[-14], method = 'pca', pcaComp = 2)
# Training Set
trs = predict(pca, trs)
trs = trs[c(2, 3, 1)]
# Test set
tes = predict(pca, tes)
tes = tes[c(2, 3, 1)]

# Fitting PCA to the training set
# install.packages('e1071')
library(e1071)
cl = svm(formula = CustomerSegmant ~ .,
         data = trs,
         type = 'C-classification',
         kernal = 'linear')

# Predicting the Test set Results
y_pred = predict(cl, newdata = tes[-3])

# Making the Confusion Matrix
cm = table(tes[, 3], y_pred)


# Visualising the Training set results
# install.packages('ElemStatLearn')
library(ElemStatLearn)
set = trs
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(cl, newdata = grid_set)
plot(set[, -3],
     main = 'Support Vector Machine (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'cyan', 'purple')))


# Visualising the Test set results
library(ElemStatLearn)
set = tes
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(cl, newdata = grid_set)
plot(set[, -3], main = 'Support Vector Machine (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'cyan', 'purple')))