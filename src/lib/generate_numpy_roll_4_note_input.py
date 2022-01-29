import numpy as np
import os

maj7_structure = np.array([0, 4, 7, 11])
m7_structure   = np.array([0, 3, 7, 10])
M7_structure   = np.array([0, 4, 7, 10])
m7b5_structure = np.array([0, 3, 6, 10])

structures = [maj7_structure, m7_structure, M7_structure, m7b5_structure]
volumes = [1/3, 2/3, 1.0]

max_base_note = 32 					# of 44 notes. THis way avary chord will have 4 notes 

numpy_dataset = np.empty([1,44])
for i in range(max_base_note):
	for structure in structures:
		for volume in volumes:
			full_structure = (structure+i)
			numpy_datapoint = volume*np.sum(np.eye(44)[full_structure],axis=0)
			numpy_dataset= np.append(numpy_dataset,[numpy_datapoint], axis = 0)

numpy_dataset = numpy_dataset[1:,:]


CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')
np.save(NUMPY_DATASET_PATH+'4_notes_roll', numpy_dataset)