from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from os import walk
import numpy as np
import time


path = "../data/arff/"

list_dataset = []
for (dirpath, dirnames, filenames) in walk(path):
    list_dataset.extend(filenames)
    break
    
list_dataset.sort()



print("----------Clustering agglomératif----------\n")
k_list = [2, 9, 20, 30, 4, 2]

link=["single", "average", "complete", "ward"]

print("-----Nombre de clusters indiqué-----\n")

for i in range(len(list_dataset)):
    print("---" + list_dataset[i] + "---")
    
    print("k = %d"%k_list[i])
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    for j in range(len(link)):
        print("linkage = %s"%link[j])
    
        time_start = time.process_time() # On regarde le temps CPU
    
        clust = cluster.AgglomerativeClustering(n_clusters=k_list[i], affinity='euclidean', linkage=link[j])
        y_pred = clust.fit_predict(data)
        
        time_stop = time.process_time() # On regarde le temps CPU
        
        print("Temps de calcul : " + str(time_stop-time_start))

        labels = clust.labels_
        print("Indice de Davies-Bouldin")
        print(metrics.davies_bouldin_score(data, labels))

        print("Coefficient de silhouette")
        print(metrics.silhouette_score(data, labels, metric='euclidean'))

        plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
        plt.show()
        
        print("\n\n")


print("-----Nombre de clusters non indiqué : méthode itérative-----\n")

link=["complete", "complete", "ward", "ward", "ward", "single"]

for i in range(len(list_dataset)):
    print("---" + list_dataset[i] + "---")
        
    print("k attendu = %d"%k_list[i])
    print("linkage = %s"%link[i])
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    db_score_list = []
    sil_score_list = []

    for k in range(2,41,1):

        clust = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=link[j])
        y_pred = clust.fit_predict(data)

        #print("Score")
        #print(clust.score(data))

        labels = clust.labels_
        # Indice de Davies-Bouldin"
        db_score = metrics.davies_bouldin_score(data, labels)
        db_score_list.append(db_score)

        # Coefficient de silhouette
        sil_score = metrics.silhouette_score(data, labels, metric='euclidean')
        sil_score_list.append(sil_score)

    plt.plot(range(len(db_score_list)), db_score_list)
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Indice de Davies Bouldin")
    plt.title("Indice de Davies Bouldin en fonction du nombre de clusters k :")
    plt.show()

    plt.plot(range(len(sil_score_list)), sil_score_list)
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Coefficient de silhouette")
    plt.title("Coefficient de silhouette en fonction du nombre de clusters k :")
    plt.show()

