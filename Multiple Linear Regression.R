# Multiple Linear Regression (MLR)

# Importing the dataset
ds = read.csv('50_Startups.csv')
# ds = ds[, 2:3] # If needing to select specific indexes

# Encoding catagorical data
ds$State = factor(ds$State, levels = c('New York', 'California', 'Florda'), labels = c(1, 2, 3))


# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$Profit, SplitRatio = 0.8)
trs = subset(ds, split == TRUE) # Training Set
tes = subset(ds, split == FALSE) # Test Set

# Fitting the MLR to the training 
lr = lm(formula = Profit ~ ., data = trs) # Equivalant to profit ~ R.D.Spend + Administration + Marketing.Spend + State

# Predicting the Test set results
y_pred = predict(lr, newdata = tes)

# Building the optimal model using Backwards Elimination
lr = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
        data = trs)
summary(lr)

lr = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(lr)

lr = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(lr)

lr = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(lr)