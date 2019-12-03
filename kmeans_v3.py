from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

list_dataset = ["connexe_petites_taches", "bruite","non_connexe", "densite_variable", "mal_separe"]
path = "./data/"

print("K_Means")
k = [30, 2, 4, 10]
name = ["connexe_petites_taches", "bruite", "densite_variable", "mal_separe"]

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
