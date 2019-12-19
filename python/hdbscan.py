import hdbscan
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


print("----------HDBSCAN----------")

for i in range(len(list_dataset)):
    print("---" + list_dataset[i] + "---")
    
    for j in range(5,50,5):
        print("min_cluster_size =  %d"%j)
        
        time_start = time.process_time() # On regarde le temps CPU
    
        dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
        data = [[x[0],x[1]] for x in dataset[0]]

        clust = hdbscan.HDBSCAN(min_cluster_size=j)
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
        
    print("--- Variation de min_samples---")
        
    min_clust_size = [20,5,5,5,15,5]
    for j in range(5,50,5):
        print("min_samples =  %d"%j)
    
        dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
        data = [[x[0],x[1]] for x in dataset[0]]

        time_start = time.process_time() # On regarde le temps CPU
        
        clust = hdbscan.HDBSCAN(min_cluster_size=min_clust_size[i], min_samples=j)
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

				
print("----------HDBSCAN----------")

min_cluster_size=[20, 5, 5, 5, 15, 5]
min_samples=[25, 30, 5, 10, 20, 15]

for i in range(len(list_dataset)):
    print("---" + list_dataset[i] + "---")
    
    time_start = time.process_time() # On regarde le temps CPU
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    clust = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size[i], min_samples=min_samples[i])
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


print("\n#--------------------- Variation de min_cluster_size et du nombre minimum de samples -----------------------#")

min_cluster_size_sil = []
min_samples_sil = []
min_cluster_size_db = []
min_samples_db = []

for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    val_db = 1000000
    val_min_samples_db = 0
    val_min_cluster_size_db = 0
    
    val_sil = -1
    val_min_samples_sil = 0
    val_min_cluster_size_sil = 0
    for min_cluster_size in np.arange(5, 50, 5):
        min_cluster_size = int(min_cluster_size)
        time_list = []
        davies_bouldin_list = []
        silhouette_score_list = []
        for samples in np.arange(5, 50, 5):
            samples = int(samples)
            dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
            data = [[x[0],x[1]] for x in dataset[0]]

            clust = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples = samples)
            y_pred = clust.fit_predict(data)

            labels = clust.labels_
            np.unique(labels)

            if len(np.unique(labels)) > 1 :
                db = metrics.davies_bouldin_score(data, labels)
                davies_bouldin_list.append(db)
                if(db < val_db):
                    val_db = db
                    val_min_cluster_size_db = min_cluster_size
                    val_min_samples_db = samples
                    
                sil = metrics.silhouette_score(data, labels, metric='euclidean')
                silhouette_score_list.append(metrics.silhouette_score(data, labels, metric='euclidean'))
                
                if (sil > val_sil):
                    val_sil = sil
                    val_min_cluster_size_sil = min_cluster_size
                    val_min_samples_sil = samples

    print("Coefficient de silhouette : " + str(val_sil))
    print("Valeur de min_cluster_size : " + str(val_min_cluster_size_sil))
    print("Valeur de min_samples : " + str(val_min_samples_sil))
    min_cluster_size_sil.append(val_min_cluster_size_sil)
    min_samples_sil.append(val_min_samples_sil)
    
    print("\nIndice de Davies-Bouldin : " + str(val_db))
    print("Valeur de min_cluster_size : " + str(val_min_cluster_size_db))
    print("Valeur de min_samples : " + str(val_min_samples_db)) 
    min_cluster_size_db.append(val_min_cluster_size_db)
    min_samples_db.append(val_min_samples_db)


print("#================ Cas Indice Davies Bouldin : Valeurs de min_cluster_size et min_samples donnés ===================#")

for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    
    time_start = time.process_time() # On regarde le temps CPU
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    clust = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_db[i], min_samples=min_samples_db[i])
    y_pred = clust.fit_predict(data)
    
    labels = clust.labels_
    
    plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
    plt.show()
    if len(np.unique(labels)) > 1 :
        print("Indice de Davies-Bouldin : " + str(metrics.davies_bouldin_score(data, labels)))
        print("Coefficient de silhouette : " + str(metrics.silhouette_score(data, labels, metric='euclidean')))

    time_stop = time.process_time() # On regarde le temps CPU
    print("Temps de calcul : " + str(time_stop-time_start))


print("#================ Cas Coefficient de silhouette : Valeurs de esp et min_samples donnés ===================#")

for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    
    time_start = time.process_time() # On regarde le temps CPU
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    clust = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_sil[i], min_samples=min_samples_sil[i])
    y_pred = clust.fit_predict(data)
    
    labels = clust.labels_
    
    plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
    plt.show()
    if len(np.unique(labels)) > 1 :
        print("Indice de Davies-Bouldin : " + str(metrics.davies_bouldin_score(data, labels)))
        print("Coefficient de silhouette : " + str(metrics.silhouette_score(data, labels, metric='euclidean')))

    time_stop = time.process_time() # On regarde le temps CPU
    print("Temps de calcul : " + str(time_stop-time_start))