# Kernel PCA

# Importing the dataset
ds = read.csv('Social_Network_Ads.csv')
ds = ds[, 3:5]

# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Purchased, SplitRatio = 0.75)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Feature scaling (if needed)
trs[, 1:2] = scale(trs[, 1:2])
tes[, 1:2] = scale(tes[, 1:2])

# install.packages('kernlab')
library(kernlab)

kpca = kpca(~ ., data = trs[-3], kernel = 'rbfdot', features = 2)
trs_pca = as.data.frame(predict(kpca, trs))
trs_pca$Purchased = trs$Purchased
tes_pca = as.data.frame(predict(kpca, tes))
tes_pca$Purchased = tes$Purchased


# Fitting Logicstic Regression to the training set
cl = glm(formula = Purchased ~ .,
         family = binomial,
         data = trs_pca)

# Predicting the Test set Results
pred = predict(cl, type = 'response', newdata = tes_pca[-3])
y_pred = ifelse(pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(tes_pca[, 3], y_pred)

# Visualising the Training set results
install.packages('ElemStatLearn')
library(ElemStatLearn)
set = trs_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(cl, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'cyan', 'purple'))


# Visualising the Test set results
library(ElemStatLearn)
set = tes_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(cl, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'cyan', 'purple'))