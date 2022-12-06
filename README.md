# clustering_examples
simple implementation of clustering methods on toy dataset

![](https://github.com/sdyy6211/clustering_examples/blob/main/animation/kmeans.gif?raw=true)
|:--:| 
| *Figure1. K-Means* |

In K-Means, 4 centroids are initialized and these 4 centroids moved to find their best places, and it can be seen that during this moving process, the loss is continuously decreasing. Finally, the algorithm stopped because all centroids do not move anymore.

![](https://github.com/sdyy6211/clustering_examples/blob/main/animation/gmm_.gif?raw=true)
|:--:| 
| *Figure2. Gaussian Mixture Model* |

In GMM, 4 Gaussian distributions are initialized and they moved their means and changed their shapes to fit the dataset. The position of distributions is determined by the means and the shape of the distributions is determined by the covariance matrix.

![](https://github.com/sdyy6211/clustering_examples/blob/main/animation/hierarchical_clustering_.gif?raw=true)
|:--:| 
| *Figure3. Hierarchical Clustering* |

In Hierarchical clustering, proximate data points will merge together first, and each merge of two data points correspond to a link of two data points in the dendrogram.

![](https://github.com/sdyy6211/clustering_examples/blob/main/animation/dbscan.gif?raw=true)
|:--:| 
| *Figure4. DBSCAN* |

The DBSCAN algorithm gradually visits each point in the dataset, and clusters points in a dense region to the same group. The radius of the circle is the epsilon parameter, and the min points parameter is the required number of points within this circle. Notice that there will be two types of points, which are center points and boundary points. The stars are central points and dots are boundary points. The algorithm will search the neighbors of central points but will not search the neighbors for boundary points.
