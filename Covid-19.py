#Covid-19 

#Importing libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

#Loading the dataset 
dataset = pd.read_csv('covid_19_data.csv')
dataset = dataset.drop(['ObservationDate','Last Update'], axis = 1)
#dataset.describe(include='all')


dataset.isnull().sum()
data_no_mv = dataset.dropna(axis =0)
data_no_mv.isnull().sum()
data_no_mv.describe(include='all')

data = data_no_mv.copy()
X = data.iloc[:, [3,4,5]].values



"""

data = data_no_mv.copy()
#sns.distplot(data['Recovered'])

#X = dataset.iloc[:, [3,4]].values 
"""


#Feature Scaling 
from sklearn import preprocessing 
data_1 = data.iloc[:,3:5]
data_1
X = preprocessing.scale(data_1)
X
#X_test = sc_X.transform(X_test)

import seaborn as sns
sns.set()
#Using Elbow method to find optimal number of K clusters 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()

#Applying Kmeans to dataset 
kmeans = KMeans(3)
kmeans.fit(X)
clustered = kmeans.fit_predict(X)
df1 = data_no_mv.copy()
df1['Clustered'] = clustered
df1
# Plotting
plt.scatter(df1['Deaths'], df1['Confirmed'], c = df1['Clustered'], cmap = 'rainbow')
plt.xlabel('Deaths')
plt.ylabel('Confirmed')
plt.show





#Applying K-means to dataset 
kmeans = KMeans(n_clusters= 3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the Clusters  

#plt.scatter(X-coordinate, Y-coordinate, s = 100, c= 'red', label = 'Cluster')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c= 'red', label = 'Confirmed')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c= 'blue', label = 'Deaths')

"""plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c= 'green', label = 'Confirmed')
   plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c= 'cyan', label = 'Careless')
   plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c= 'magenta', label = 'Sensible')
"""

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c= 'black', label = 'Centroids')
plt.title('Covid-19')
plt.xlabel('Deaths')
plt.ylabel('Confirmed')
plt.legend()
plt.show()

plt.scatter(data['Recovered'], data['Confirmed'])
plt.xlabel('RECOVERED')
plt.ylabel('CONFIRMED')
plt.show
