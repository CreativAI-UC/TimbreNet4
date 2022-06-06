import os
import numpy as np
from sklearn.decomposition import PCA

CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATASET_PATH = os.path.join(os.path.dirname(CODE_FOLDER),'datasets/numpyDatasets')

triad_train_augmentd = np.load(DATASET_PATH+'/3_notes_roll_train_augmented.npy')
triad_train = np.load(DATASET_PATH+'/3_notes_roll_train.npy')
triad_val   = np.load(DATASET_PATH+'/3_notes_roll_val.npy')
tetrad_val  = np.load(DATASET_PATH+'/4_notes_roll.npy')
random_val  = np.load(DATASET_PATH+'/3_random_notes_roll.npy')

def print_explained_variance(dataset):
	print("ANALYSIS: ")
	pca = PCA(n_components=45)
	principalComponents = pca.fit_transform(dataset)
	cumulate = 0
	for idx, i in enumerate(pca.explained_variance_ratio_):
		cumulate += i
		print("{:02d}".format(idx+1), end = ':   ')
		print("{:.6f}".format(i), end = '   ')
		print("{:.6f}".format(cumulate))


print_explained_variance(triad_train_augmentd)
print_explained_variance(triad_train)
print_explained_variance(triad_val)
print_explained_variance(tetrad_val)
print_explained_variance(random_val)

'''
80%
25






'''