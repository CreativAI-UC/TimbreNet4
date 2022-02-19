from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf



	





def model_get_chord_latent_pair(run_path, data, verbose=False):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    # MODEL
    TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])
    TN_VAE.load_weights(os.path.join(run_path, 'weights/weights.h5'))
    print('MODEL LOADED')

    
    latent, _  = TN_VAE.encoder(data)

    return data, latent



def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    THREE_NOTES_DATA_PATH   = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/3_notes_roll.npy')
    FOUR_NOTES_DATA_PATH    = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/4_notes_roll.npy')

    data_3_notes = np.load(THREE_NOTES_DATA_PATH)
    data_4_notes = np.load(FOUR_NOTES_DATA_PATH)


    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
                data_3, latent_3 = model_get_chord_latent_pair(RUN_PATH, data_3_notes)
                data_4, latent_4 = model_get_chord_latent_pair(RUN_PATH, data_4_notes)

                np.save(RUN_PATH+'/3_notes_chords', data_3)
                np.save(RUN_PATH+'/3_notes_latent', latent_3)

                np.save(RUN_PATH+'/4_notes_chords', data_4)
                np.save(RUN_PATH+'/4_notes_latent', latent_4)






	

    




        
if __name__=="__main__":
    main()