import numpy as np
import os
#from random import shuffle

m1 = np.array([0, 4, 7])
m2 = np.array([0, 3, 8])
m3 = np.array([0, 5, 9])


n1 = np.array([0, 3, 7])
n2 = np.array([0, 4, 9])
n3 = np.array([0, 5, 8])

a = np.array([0, 4, 8])

d1 = np.array([0, 3, 6])
d2 = np.array([0, 3, 9])
d3 = np.array([0, 6, 9])

structures = [m1,m2,m3,n1,n2,n3,a,d1,d2,d3]
volumes = [1/3, 2/3, 1.0]

max_base_note = 36 					# of 45 notes. THis way avary chord will have 3 notes 

numpy_dataset = np.empty([1,45])
for i in range(max_base_note):
	for structure in structures:
		for volume in volumes:
			full_structure = (structure+i)
			numpy_datapoint = volume*np.sum(np.eye(45)[full_structure],axis=0)
			numpy_dataset= np.append(numpy_dataset,[numpy_datapoint], axis = 0)
			#print(numpy_datapoint)

numpy_dataset = numpy_dataset[1:,:]

#  972
#  108
# 1080


np.random.shuffle(numpy_dataset)
numpy_train = numpy_dataset[0:972,:]
numpy_val = numpy_dataset[972:,:]


CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')
np.save(NUMPY_DATASET_PATH+'triad_dataset', numpy_dataset)
print(np.shape(numpy_dataset))

np.save(NUMPY_DATASET_PATH+'triad_train', numpy_train)
print(np.shape(numpy_train))
np.save(NUMPY_DATASET_PATH+'triad_val', numpy_val)
print(np.shape(numpy_val))
