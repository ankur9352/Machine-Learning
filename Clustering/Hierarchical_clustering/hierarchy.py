import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
data=pd.read_csv("Mall_Customers.csv")
x=data.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c="red",label="Cluster 1")
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c="blue",label="Cluster 2")
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c="green",label="Cluster 3")
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c="cyan",label="Cluster 4")
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c="magenta",label="Cluster 5")
plt.title("Clusters of clients")
plt.xlabel("Annual Income (K$")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
