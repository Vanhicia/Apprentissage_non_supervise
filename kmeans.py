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
    
print("#==================== Méthode : K_Means ======================#")
k_list = [2, 9, 20, 30, 4, 2]

print("#================ Nombre de clusters donné ===================#")

for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    
    time_start = time.process_time() # On regarde le temps CPU
    
    dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    kmeans = cluster.KMeans(n_clusters=k_list[i], init='k-means++')
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    labels = kmeans.labels_
    
    plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
    plt.show()
    
    print("Indice de Davies-Bouldin : " + str(metrics.davies_bouldin_score(data, labels)))
    print("Coefficient de silhouette : " + str(metrics.silhouette_score(data, labels, metric='euclidean')))

    time_stop = time.process_time() # On regarde le temps CPU
    print("Temps de calcul : " + str(time_stop-time_start))

print("\n#------------------ Nombre de clusters NON donné ------------------#")  
k_max = [x+15 for x in k_list]
for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    score = []
    db = []
    sil = []
    temps = []
    for k in range(2, k_max[i]):
        time_start = time.process_time() # On regarde le temps CPU

        dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
        data = [[x[0],x[1]] for x in dataset[0]]

        kmeans = cluster.KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(data)
        y_pred = kmeans.predict(data)
        labels = kmeans.labels_
        db.append(metrics.davies_bouldin_score(data, labels))
        sil.append(metrics.silhouette_score(data, labels, metric='euclidean'))
        time_stop = time.process_time() # On regarde le temps CPU
        temps.append(time_stop-time_start)
    print("Indice de Davies-Bouldin : " + str(db))
    print("Coefficient de silhouette : " + str(sil))
    print("Temps de calcul : " + str(temps))

    plt.plot(range(2, k_max[i]), db)
    plt.xlabel("Valeur de k")
    plt.ylabel("Indice de Davies-Bouldin")
    plt.title("Indice de Davies-BOuldin en fonction de k :")
    plt.show()
    
    plt.xlabel("Valeur de k")
    plt.ylabel("Coefficient de silhouette")
    plt.title("Coefficient de silhouette en fonction de k :")
    plt.plot(range(2, k_max[i]), sil)
    plt.show()
    
    plt.xlabel("Valeur de k")
    plt.ylabel("Temps")
    plt.title("Temps en fonction de k :")
    plt.plot(range(2, k_max[i]), temps)
    plt.show()