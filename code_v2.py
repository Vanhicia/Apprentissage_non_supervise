from scipy.io import arff
from sklearn import cluster
import numpy as np
from matplotlib import pyplot as plt

list_dataset = ["connexe_petites_taches", "bruite","non_connexe", "densite_variable", "mal_separe"]
path = "./data/"

print("K_Means")
k = [30, 2, 4, 10]
name = ["connexe_petites_taches", "bruite", "densite_variable", "mal_separe"]

for i in range(len(name)):
    
    dataset = arff.loadarff(open(path + name[i] + '.arff', 'r'))
    print(dataset)
    data = [[x[0],x[1]] for x in dataset[0]]
    kmeans = cluster.KMeans(n_clusters=k[i], init='k-means++')
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    print("Score")
    print(kmeans.score(data))
    plt.scatter(dataset[0]['x'], dataset[0]['y'], c=y_pred, cmap="tab20")
    plt.show()
    
    






