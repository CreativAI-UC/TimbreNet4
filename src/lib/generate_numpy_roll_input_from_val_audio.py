from src.models.pianorollModel import PianoRollModel

import os
import numpy as np
import tensorflow as tf

TN_VAE = PianoRollModel(4,2,32)

CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NUMPY_DATASET_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/')
VAL_DATA_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/audioPianoTriadDataset/audio_val/*')
VAL_FILENAMES  = tf.data.Dataset.list_files(VAL_DATA_PATH, shuffle=False)
VAL_DATASET    = VAL_FILENAMES.map(TN_VAE.pre_process_filename_to_roll).batch(1)


NUMPY_DATASET = np.empty([1,44])

for batch_number, (val_data_point_in, val_data_point_out) in enumerate(VAL_DATASET):
    numpy_datapoint = val_data_point_in.numpy()
    NUMPY_DATASET= np.append(NUMPY_DATASET,numpy_datapoint, axis = 0)

NUMPY_DATASET = NUMPY_DATASET[1:,:]

np.save(NUMPY_DATASET_PATH+'validation', NUMPY_DATASET)
