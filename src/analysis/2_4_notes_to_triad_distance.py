from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf


def get_parent_triads(data):
	n_examples = np.shape(data)[0]
	parent_1 = np.copy(data)
	parent_2 = np.flip(np.copy(data), axis=1)

	for i in range(n_examples):
		for j in range(44):
			if parent_1[i,j] != 0:
				parent_1[i,j] = 0
				break
	for i in range(n_examples):
		for j in range(44):
			if parent_2[i,j] != 0:
				parent_2[i,j] = 0
				break

	parent_2 = np.flip(parent_2, axis=1)

	return parent_1, parent_2
	





def model_get_4_note_to_triad_distance_proportion(run_path, data, verbose=False):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    # MODEL
    TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])
    TN_VAE.load_weights(os.path.join(run_path, 'weights/weights.h5'))
    print('MODEL LOADED')

    data_4_note, data_parent_triad_1, data_parent_triad_2 = data
    
    mu_data, _  = TN_VAE.encoder(data_4_note)
    mu_parent_1, _  = TN_VAE.encoder(data_parent_triad_1)
    mu_parent_2, _  = TN_VAE.encoder(data_parent_triad_2)

    dist_1 = np.linalg.norm(mu_data-mu_parent_1, axis = 1)
    dist_2 = np.linalg.norm(mu_data-mu_parent_2, axis = 1)

    proportion = np.abs(dist_1-dist_2)/(dist_1+dist_2)
    proportion_mu = np.mean(proportion)
    proportion_std = np.std(proportion)


    result_str = "\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM'])
    result_str += "\n\tMEAN PROPORTION DISTANCE: {:.5f}".format(proportion_mu)
    result_str += "\n\tSTD PROPORTION DISTANCE: {:.5f}".format(proportion_std)

    return result_str



def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    FOUR_NOTES_DATA_PATH   = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/4_notes_roll.npy')

    data_4_note = np.load(FOUR_NOTES_DATA_PATH)
    data_parent_triad_1, data_parent_triad_2 = get_parent_triads(data_4_note)
    data = (data_4_note, data_parent_triad_1, data_parent_triad_2)


    result_txt          = ''

    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
                result_txt += model_get_4_note_to_triad_distance_proportion(RUN_PATH, data)



    with open(os.path.join(CODE_PATH, 'analysis_results/2_4_notes_to_triad_distance/result.txt'), 'w') as text_file:
        text_file.write(result_txt)

	

    




        
if __name__=="__main__":
    main()