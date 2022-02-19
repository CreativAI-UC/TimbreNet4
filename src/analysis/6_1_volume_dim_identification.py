from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree



def vol_dim_identification(run_path, verbose=False):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    # MODEL
    TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])
    TN_VAE.load_weights(os.path.join(run_path, 'weights/weights.h5'))
    print('MODEL LOADED')

    # THREE NOTES VOLUME DISTANCE

    m = np.array([0, 4, 7])
    n = np.array([0, 3, 7])
    a = np.array([0, 4, 8])
    d = np.array([0, 3, 6])

    structures = [m,n,a,d]
    volumes = [1/3, 2/3, 1.0]

    max_base_note = 36                  # of 44 notes. THis way avary chord will have 3 notes 


    p_dataset = np.empty([1,44])
    m_dataset = np.empty([1,44])
    f_dataset = np.empty([1,44])


    for i in range(max_base_note):
        for structure in structures:
            full_structure = (structure+i)
            
            p_datapoint = volumes[0]*np.sum(np.eye(44)[full_structure],axis=0)
            m_datapoint = volumes[1]*np.sum(np.eye(44)[full_structure],axis=0)
            f_datapoint = volumes[2]*np.sum(np.eye(44)[full_structure],axis=0)

            p_dataset   = np.append(p_dataset,[p_datapoint], axis = 0)
            m_dataset   = np.append(m_dataset,[m_datapoint], axis = 0)
            f_dataset   = np.append(f_dataset,[f_datapoint], axis = 0)

    p_dataset = p_dataset[1:,:]
    m_dataset = m_dataset[1:,:]
    f_dataset = f_dataset[1:,:]

    p_latent, _  = TN_VAE.encoder(p_dataset)
    m_latent, _  = TN_VAE.encoder(m_dataset)
    f_latent, _  = TN_VAE.encoder(f_dataset)

    p_latent_std = np.std(p_latent, axis=0)
    m_latent_std = np.std(m_latent, axis=0)
    f_latent_std = np.std(f_latent, axis=0)

    p_latent_mu = np.mean(p_latent, axis=0)
    m_latent_mu = np.mean(m_latent, axis=0)
    f_latent_mu = np.mean(f_latent, axis=0)

    latent_mu = np.stack((p_latent_mu, m_latent_mu, f_latent_mu))
    latent_std = np.std(latent_mu, axis=0)

    result_txt = str("\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}\n".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM']))

    result_txt += str("\np: ")
    result_txt += str((-p_latent_std).argsort()[-4:][::-1])
    result_txt += str("\nm: ")
    result_txt += str((-m_latent_std).argsort()[-4:][::-1])
    result_txt += str("\nf: ")
    result_txt += str((-f_latent_std).argsort()[-4:][::-1])
    result_txt += str("\n\n   ")
    result_txt += str(latent_std.argsort()[-4:][::-1])


    result_txt += str(" ")
    
    return result_txt



                







def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    result_txt = ''


    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
    
                result_txt += vol_dim_identification(RUN_PATH)

    with open(os.path.join(CODE_PATH, 'analysis_results/6_1_volume_dim_identification/result.txt'), 'w') as text_file:
        text_file.write(result_txt)






	

    




        
if __name__=="__main__":
    main()