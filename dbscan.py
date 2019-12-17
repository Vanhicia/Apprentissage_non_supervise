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
		
print("\n#=================== Valeurs de esp et min_samples NON donné ===================#")  

print("\n#--------------------- Variation de epsilon -----------------------#")
for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    time_list = []
    davies_bouldin_list = []
    silhouette_score_list = []
    value_x = []

    for esp in np.arange(0.01, 1, 0.05):
        time_start = time.process_time() # On regarde le temps CPU
        
        dataset = arff.loadarff(open(path + list_dataset[i], 'r'))
        data = [[x[0],x[1]] for x in dataset[0]]

        dbscan = cluster.DBSCAN(eps = esp, min_samples = min_samples[i])
        y_pred = dbscan.fit_predict(data)

        labels = dbscan.labels_
        np.unique(labels)

        if len(np.unique(labels)) > 1 :
            value_x.append(esp)
            davies_bouldin_list.append(metrics.davies_bouldin_score(data, labels))
            silhouette_score_list.append(metrics.silhouette_score(data, labels, metric='euclidean'))
        time_stop = time.process_time() # On regarde le temps CPU
        time_list.append(time_stop-time_start)
        
    print("Coefficient de silhouette : " + str(silhouette_score_list))
    print("Indice de Davies-Bouldin : " + str(davies_bouldin_list))
    print("Temps de calcul : " + str(time_list))
    
    plt.plot(value_x, silhouette_score_list)
    plt.xlabel("Valeur de epsilon")
    plt.ylabel("Coefficient de silhouette")
    plt.title("Coefficient de silhouette en fonction de la valeur d'epsilon")
    plt.show()

    plt.plot(value_x, davies_bouldin_list)
    plt.xlabel("Valeur de epsilon")
    plt.ylabel("Indice de Davies Bouldin")
    plt.title("Indice de Davies Bouldin en fonction de la valeur d'epsilon")
    plt.show()
    
    plt.plot(np.arange(0.01, 1, 0.05), time_list)
    plt.xlabel("Valeur de epsilon")
    plt.ylabel("Temps")
    plt.title("Temps en fonction de la valeur d'epsilon")
    plt.show()
		
print("\n#--------------------- Variation du nombre minimum de samples -----------------------#")
min_samples_range = [10,35,10,20,40]

"""
for min_samples in range(1,l[i]):
		time_start = time.process_time() # On regarde le temps CPU
		dbscan = cluster.DBSCAN(eps = esp, min_samples = min_samples)
		y_pred = dbscan.fit_predict(data)

		labels = dbscan.labels_
		np.unique(labels)

		if len(np.unique(labels)) > 1 :
				x.append(min_samples)
				davies_bouldin = metrics.davies_bouldin_score(data, labels)
				davies_bouldin_list.append(davies_bouldin)

				silhouette_score = metrics.silhouette_score(data, labels, metric='euclidean')
				silhouette_score_list.append(silhouette_score)
				print("------------- ESP  = " + str(esp) + "--------------")
				print("------------- MIN SAMPLES  = " + str(min_samples) + "--------------")
				plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
				plt.show()

		time_stop = time.process_time() # On regarde le temps CPU
		time_list.append(time_stop-time_start)
"""
