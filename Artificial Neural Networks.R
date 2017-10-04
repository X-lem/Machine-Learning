# Artificial Neural Network (ANN)

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

# Feature scaling
trs[-11] = scale(trs[-11])
tes[-11] = scale(tes[-11])


# Fitting ANN to the training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
cl = h2o.deeplearning(y = 'Exited', training_frame = as.h2o(trs),
                      activation = 'Rectifier', hidden = c(6, 6),
                      epochs = 100, train_samples_per_iteration = -2)


# Predicting the Test set Results
prob_pred = h2o.predict(cl, newdata = as.h2o(tes[-11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(tes[, 11], y_pred)

h2o.shutdown()