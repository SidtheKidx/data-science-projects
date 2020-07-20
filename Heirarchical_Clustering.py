#HC Clustering 

#IMPORTING LIBRARIES 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
sns.set()

#Loading the dataset 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values 

#Using the Dendrogram to find the optimal no. of clusters 
import scipy.cluster.hierarchy as sch 
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting HC to the dataset 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters only for 2 dimensions 
#plt.scatter(X-coordinate, Y-coordinate, s = 100, c= 'red', label = 'Cluster')
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c= 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c= 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c= 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c= 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c= 'magenta', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k)$')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


"""
#Visualizing the clusters in 3D 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Axes3D.scatter(xs=0, ys=0, zs=0, zdir='z', s=20, c='green', depthshade=True, *args, **kwargs)
plt.legend()
plt.show()
"""


