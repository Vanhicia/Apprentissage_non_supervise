from scipy.io import arff
path = "./data/"
dataset_bruite = arff.loadarff(open(path + 'bruite.arff', 'r'))
print(dataset_bruite)
data_bruite = dataset_bruite[0]['CLASS']


dataset_connexe = arff.loadarff(open(path + 'connexe_petites_taches.arff', 'r'))
data_connexe = dataset_connexe[0]['CALSS']

dataset_non_connexe = arff.loadarff(open(path + 'non_connexe.arff', 'r'))
data_non_connexe = dataset_non_connexe[0]['CALSS']


dataset_densite_variable = arff.loadarff(open(path + 'non_densite_variable.arff', 'r'))
data_densite_variable = dataset_densite_variable[0]['CALSS']

dataset_densite_non_variable = dataset_non_connexe
data_densite_variable = dataset_densite_non_variable[0]['CALSS']