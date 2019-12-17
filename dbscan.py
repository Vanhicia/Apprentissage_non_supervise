from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from os import walk
import numpy as np
import time

path = "./data/arff/"

list_dataset = []
for (dirpath, dirnames, filenames) in walk(path):
    list_dataset.extend(filenames)
    break
    

print("#==================== Méthode : DBSCAN ======================#")
esp = [0.06,0.31,0.81,0.71,0.96,0.06]
min_samples = [5,34,2,19,11,2]

print("#================ Valeurs de esp et min_samples donnés ===================#")

for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    
    time_start = time.process_time() # On regarde le temps CPU
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    dbscan = cluster.DBSCAN(eps=esp[i], min_samples=min_samples[i])
    y_pred = dbscan.fit_predict(data)
    
    labels = dbscan.labels_
    
    plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
    plt.show()
    if len(np.unique(labels)) > 1 :
        print("Indice de Davies-Bouldin : " + str(metrics.davies_bouldin_score(data, labels)))
        print("Coefficient de silhouette : " + str(metrics.silhouette_score(data, labels, metric='euclidean')))

    time_stop = time.process_time() # On regarde le temps CPU
    print("Temps de calcul : " + str(time_stop-time_start))