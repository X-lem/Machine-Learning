# Hierarchical Clustering

# Importing the dataset
ds = read.csv('Mall_Customers.csv')
X = ds[, 4:5] # If needing to select specific indexes

# Using the dendrogram to find the optimal number of clusters
dn = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dn, main = paste('Dendrogram'), xlab = 'Customers', ylab = "Euclidean Distances")


# Fitting the hierarchical Clustering to the dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(ds,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')