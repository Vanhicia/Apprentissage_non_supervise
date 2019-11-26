import numpy as np
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.cluster import KMeans

path = "./data/"


dataset_bruite = arff.loadarff(open(path + 'bruite.arff', 'r'))
data_bruite = dataset_bruite[0]



#print(val)
#plt.scatter(data_bruite['x'], data_bruite['y'])
#plt.show()

dataset_convexe = arff.loadarff(open(path + 'connexe_petites_taches.arff', 'r'))
data_convexe = dataset_convexe[0]

val = [[x[0],x[1]] for x in data_convexe]
 
dataset_non_convexe = arff.loadarff(open(path + 'non_connexe.arff', 'r'))
data_non_convexe = dataset_non_convexe[0]['class']


dataset_densite_variable = arff.loadarff(open(path + 'densite_variable.arff', 'r'))
data_densite_variable = dataset_densite_variable[0]['class']

dataset_densite_non_variable = dataset_non_convexe
data_densite_non_variable = dataset_densite_non_variable[0]['class']

print("K_Means")
kmeans = KMeans(n_clusters=30, init='k-means++')
kmeans.fit(val)
y_pred = kmeans.predict(val)
print("Score")
print(kmeans.score(val))
plt.scatter(data_convexe['x'], data_convexe['y'], c=y_pred)
plt.show()
