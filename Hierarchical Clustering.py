# Hierarchical Clustering

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv("Mall_Customers.csv")
X = ds.iloc[:, [3,4]].values # [Rows, Columns] : = take all


# Finding the optimal number of clusters (using dendrogram)
import scipy.cluster.hierarchy as sch
dn = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()


# Fitting the hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_ac = ac.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_ac == 2, 0], X[y_ac == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_ac == 3, 0], X[y_ac == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_ac == 4, 0], X[y_ac == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()