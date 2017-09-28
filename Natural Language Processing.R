# Natural Language Processing

# Importing the dataset
ds = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the text
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(ds$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # toLowercase
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords()) # Remove unnessisary words
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating a bag word model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999) # Keeps the top 99.9% of words.
ds_class = as.data.frame(as.matrix(dtm))
ds_class$Liked = ds$Liked

##### Using the Random Forest Classification #####

# Encoding the target feature as factor
ds_class$Liked = factor(ds_class$Liked, levels = c(0, 1))

# Splitting the dataset into a training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds_class$Liked, SplitRatio = 0.8)
trs = subset(ds_class, split == TRUE) # Training Set
tes = subset(ds_class, split == FALSE) # Test Set

# Fitting the Random Forest Classification to the training set
# install.packages('randomForest')
library(randomForest)
cl = randomForest(x = trs[-692],
                  y = trs$Liked,
                  ntree = 10)

# Predicting the Test set Results
y_pred = predict(cl, newdata = tes[-692])

# Making the Confusion Matrix
cm = table(tes[, 692], y_pred)
