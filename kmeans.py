from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

import numpy as np
import time

list_dataset = ["convexe_petites_taches", "bruite_convexe","non_convexe", "densite_variable", "mal_separe"]
path = "./Apprentissage_non_supervise/data/"

print("#==================== Méthode : K_Means ======================#")
k = [30, 2, 2, 4, 10]

print("#================ Nombre de clusters donné ===================#")

for i in range(len(list_dataset)):
    print("\n#----------------- " + list_dataset[i] + " --------------------#")
    
		time_start = time.process_time() # On regarde le temps CPU
    
		dataset = arff.loadarff(open(path + list_dataset[i] + '.arff', 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    kmeans = cluster.KMeans(n_clusters=k[i], init='k-means++')
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    labels = kmeans.labels_
    
		plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred, cmap="tab20")
    plt.show()
    
		print("Score : "+ str(kmeans.score(data)))
    print("Indice de Davies-Bouldin : " + str(metrics.davies_bouldin_score(data, labels)))
    print("Coefficient de silhouette : " + str(metrics.silhouette_score(data, labels, metric='euclidean')))
		
    time_stop = time.process_time() # On regarde le temps CPU
    print("Temps de calcul : " + str(time_stop-time_start))

print("\n#------------------ Nombre de clusters NON donné ------------------#")  