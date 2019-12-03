from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

list_dataset = ["convexe_petites_taches", "bruite","non_convexe", "densite_variable", "mal_separe"]
path = "./data/"

print("K_Means")
k = [30, 2, 4, 10]
name = ["convexe_petites_taches", "bruite", "densite_variable", "mal_separe"]

for i in range(4):
    print(name[i])
    dataset = arff.loadarff(open(path + name[i] + '.arff', 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    kmeans = cluster.KMeans(n_clusters=k[i], init='k-means++')
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    print("Score")
    print(kmeans.score(data))
    
    labels = kmeans.labels_
    print("Indice de Davies-Bouldin")
    print(metrics.davies_bouldin_score(data, labels))
    
    print("Coefficient de silhouette")
    print(metrics.silhouette_score(data, labels, metric='euclidean'))

    plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred)
    plt.show()


dataset_bruite = arff.loadarff(open(path + 'bruite.arff', 'r'))
data_bruite = dataset_bruite[0]


"""
#print(data)
#plt.scatter(data_bruite['x'], data_bruite['y'])
#plt.show()

dataset_convexe = arff.loadarff(open(path + 'convexe_petites_taches.arff', 'r'))
data_convexe = dataset_convexe[0]

data = [[x[0],x[1]] for x in data_convexe]
 
dataset_non_convexe = arff.loadarff(open(path + 'non_convexe.arff', 'r'))
data_non_convexe = dataset_non_convexe[0]['class']


dataset_densite_variable = arff.loadarff(open(path + 'densite_variable.arff', 'r'))
data_densite_variable = dataset_densite_variable[0]['class']

dataset_densite_non_variable = dataset_non_convexe
data_densite_non_variable = dataset_densite_non_variable[0]['class']

print("K_Means")
kmeans = KMeans(n_clusters=30, init='k-means++')
kmeans.fit(data)
y_pred = kmeans.predict(data)
print("Score")
print(kmeans.score(data))
plt.scatter(data_convexe['x'], data_convexe['y'], c=y_pred, cmap="tab20")
plt.show()
"""
