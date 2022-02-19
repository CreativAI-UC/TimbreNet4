import numpy as np
import os
#from random import shuffle

m = np.array([0, 4, 7])
n = np.array([0, 3, 7])
a = np.array([0, 4, 8])
d = np.array([0, 3, 6])

structures = [m,n,a,d]
volumes = [1/3, 2/3, 1.0]

max_base_note = 36 					# of 44 notes. THis way avary chord will have 3 notes 

numpy_dataset = np.empty([1,44])
for i in range(max_base_note):
	for structure in structures:
		for volume in volumes:
			full_structure = (structure+i)
			numpy_datapoint = volume*np.sum(np.eye(44)[full_structure],axis=0)
			numpy_dataset= np.append(numpy_dataset,[numpy_datapoint], axis = 0)
			#print(numpy_datapoint)

numpy_dataset = numpy_dataset[1:,:]


np.random.shuffle(numpy_dataset)
numpy_train = numpy_dataset[0:345,:]
numpy_val = numpy_dataset[345:,:]


CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')
np.save(NUMPY_DATASET_PATH+'3_notes_roll', numpy_dataset)
print(np.shape(numpy_dataset))

np.save(NUMPY_DATASET_PATH+'3_notes_roll_train', numpy_train)
print(np.shape(numpy_train))
np.save(NUMPY_DATASET_PATH+'3_notes_roll_val', numpy_val)
print(np.shape(numpy_val))
