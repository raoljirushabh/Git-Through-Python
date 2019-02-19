# import the required libraries

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np

#importing IRIS data
iris = datasets.load_iris()

#we create a dataframe using only the features of our iris data
x = pd.DataFrame(iris.data)
print (x)
# assign a new column name
x.columns = ['sepal_length','sepal_width','petal_length','petal_width']

#create a dataframe for the target variable which we can use to compare the output
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

#we use kmeans clustering to cluster our data in 3 groups
model = KMeans(n_clusters = 3)
model.fit(x)
print (model.labels_)

#visualise our cluster
colormap = np.array(['red','blue','green'])
plt.scatter(x.petal_length, x.petal_width, c = colormap[model.labels_],s = 40)
#compare tagets vs clusters
plt.title('Kmeans Cluster')
plt.show()

#plot our target value
plt.scatter(x.petal_length,x.petal_width, c= colormap[y.Targets],s = 40 )
plt.title('Actual Cluster')
plt.show()

#our model grouped the data based on the feature values. and this grouping is almost same as our target values.
