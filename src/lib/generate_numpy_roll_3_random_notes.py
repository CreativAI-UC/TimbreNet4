import numpy as np
import os


volumes = [1/3, 2/3, 1.0]
notes = 44

numpy_dataset = np.empty([1,44])

for i in range(1000):
	for volume in volumes:
		structure = np.random.randint(notes, size=(3))
		numpy_datapoint = volume*np.sum(np.eye(44)[structure],axis=0)
		numpy_dataset= np.append(numpy_dataset,[numpy_datapoint], axis = 0)

numpy_dataset = numpy_dataset[1:,:]

CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')
np.save(NUMPY_DATASET_PATH+'random_notes_roll', numpy_dataset)