from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
import time

list_dataset = ["convexe_petites_taches", "bruite_convexe","non_convexe", "densite_variable", "mal_separe"]
path = "./data/"

print("DBSCAN")

esp = [0.3,0.08,0.3,0.4,0.3]
min_samples = [2,30,3,2,3]
for i in range(len(list_dataset)):
    print(list_dataset[i])
    time_start = time.process_time() # On regarde le temps CPU
    dataset = arff.loadarff(open(path + list_dataset[i] + '.arff', 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    dbscan = cluster.DBSCAN(eps=esp[i], min_samples=min_samples[i])
    y_pred= dbscan.fit_predict(data)
    
    labels = dbscan.labels_
    np.unique(labels)
    print("Labels")
    if len(np.unique(labels)) >1 :
        print("Indice de Davies-Bouldin")
        print(metrics.davies_bouldin_score(data, labels))

        print("Coefficient de silhouette")
        print(metrics.silhouette_score(data, labels, metric='euclidean'))

    plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
    plt.show()
    time_stop = time.process_time() # On regarde le temps CPU
    print("Temps de calcul : " + str(time_stop-time_start))

    