import numpy as np
import os


volumes = [1/3, 2/3, 1.0]
notes = 45

numpy_dataset = np.empty([1,45])

for i in range(1000):
	for volume in volumes:
		structure = np.random.choice(range(notes), 3, replace=False)
		numpy_datapoint = volume*np.sum(np.eye(45)[structure],axis=0)
		numpy_dataset= np.append(numpy_dataset,[numpy_datapoint], axis = 0)

numpy_dataset = numpy_dataset[1:,:]

CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')
np.save(NUMPY_DATASET_PATH+'3_random_notes_roll', numpy_dataset)