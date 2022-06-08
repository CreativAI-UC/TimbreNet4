import numpy as np
import os

CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')


numpy_train_augmented = np.empty([1,45])

with open(NUMPY_DATASET_PATH+'triad_train.npy', 'rb') as f:
	data = np.load(f)
	data = np.repeat(data,1000,axis=0)
	data_nosie =  np.random.normal(0, 1/(6*3*2), np.shape(data))
	data = data + data_nosie
	np.save(NUMPY_DATASET_PATH+'triad_train_augmented', data)
	print(np.shape(data))

