from scipy.io import arff
path = "./data/"
dataset_bruite = arff.loadarff(open(path + 'bruite.arff', 'r'))
print(dataset_bruite)
data_bruite = dataset_bruite[0]['CLASS']


dataset_connexe = arff.loadarff(open(path + 'connexe_petites_taches.arff', 'r'))

data_connexe = dataset_connexe[0]['class']

dataset_non_connexe = arff.loadarff(open(path + 'non_connexe.arff', 'r'))
data_non_connexe = dataset_non_connexe[0]['class']


dataset_densite_variable = arff.loadarff(open(path + 'densite_variable.arff', 'r'))
print(dataset_densite_variable)
data_densite_variable = dataset_densite_variable[0]['class']

dataset_densite_non_variable = dataset_non_connexe
data_densite_non_variable = dataset_densite_non_variable[0]['class']

print("K_Means")
kmeans = cluster.KMeans(n_clusters=30, init='k-means++')
kmeans.fit(data_connexe)
kmeans.predict(data_connexe)
