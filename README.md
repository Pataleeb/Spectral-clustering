# Spectral-clustering
Both k-means clustering and spectral clustering are popular methods for grouping data, but they differ in their approach and types of clusters they can effectively identify.

K-means clustering is a centroid-based algorithm. It initializes k centroids and assign each data point to the nearest centoid based on a distance metric and then recalculates the centroids as the mean of the assigned points. 

Spectral clustering uses the eigenvalues and eigenvectors of a similarity matrix constructed using the data. By performing eigen decomposition, spectral clustering projects the data into a lower-dimensional space where the structure of the data is more apparent, and then applies a clustering method on these projections. 

In this code, we will study a political blog dataset by Adamic and Natalie Glance. "The political blogosphere and the 2004 US Election". It is assumed that blog-site with the same political orientation are more likely to link to each other and form a "community". 

The dataset nodes.text contains a graph with n=1490 vertices (nodes) corresponding to political blogs.

The dataset edges.txt contains edges between the vertices. 

We will treat the network as an unidirected graph.


