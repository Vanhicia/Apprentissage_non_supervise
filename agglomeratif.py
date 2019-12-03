from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

list_dataset = ["convexe_petites_taches", "bruite_convexe", "densite_variable", "mal_separe", "non_convexe"]
path = "./data/"

print("----------Clustering agglomératif----------")
k_list = [30, 2, 4, 10, 2]

link=["single", "average", "complete", "ward"]

# nombre de cluters k indiqué
print("-----Nombre de clusters indiqué-----")

for i in range(len(list_dataset)):
    print("Dataset %d"%i)
    print(list_dataset[i])
    
    print("k = %d"%k_list[i])
    
    dataset = arff.loadarff(open(path + list_dataset[i] + '.arff', 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    for j in range(len(link)):
        print("linkage = %s"%link[j])
    
        clust = cluster.AgglomerativeClustering(n_clusters=k_list[i], affinity='euclidean', linkage=link[j])
        y_pred = clust.fit_predict(data)
        
        #print("Score")
        #print(clust.score(data))

        labels = clust.labels_
        print("Indice de Davies-Bouldin")
        print(metrics.davies_bouldin_score(data, labels))

        print("Coefficient de silhouette")
        print(metrics.silhouette_score(data, labels, metric='euclidean'))

        plt.scatter((dataset[0])['x'], (dataset[0])['y'], c=y_pred)
        plt.show()
        
        
# nombre de cluters non indiqué
print("-----Nombre de clusters non indiqué : méthode itérative-----")

for i in range(len(list_dataset)):
    print("Dataset %d"%i)
    print(list_dataset[i])
    
    if (list_dataset[i]=="non_convexe"):
        print("dataset non convexe")
    else:
        print("dataset convexe")
        
    print("k attendu = %d"%k_list[i])
    
    dataset = arff.loadarff(open(path + list_dataset[i] + '.arff', 'r'))
    data = [[x[0],x[1]] for x in dataset[0]]

    for j in range(len(link)):
        print("linkage = %s"%link[j])
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
