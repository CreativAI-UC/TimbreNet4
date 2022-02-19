from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree



def vol_avg_distance(run_path, verbose=False):
    
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


    n_3 = 0
    pm_dist_3 = 0
    mf_dist_3 = 0
    pf_dist_3 = 0
    
    for i in range(max_base_note):
        for structure in structures:
            full_structure = (structure+i)

            p_datapont = volumes[0]*np.sum(np.eye(44)[full_structure],axis=0).reshape((1,44))
            m_datapont = volumes[1]*np.sum(np.eye(44)[full_structure],axis=0).reshape((1,44))
            f_datapont = volumes[2]*np.sum(np.eye(44)[full_structure],axis=0).reshape((1,44))

            p, _  = TN_VAE.encoder(p_datapont)
            m, _  = TN_VAE.encoder(m_datapont)
            f, _  = TN_VAE.encoder(f_datapont)

            pm_dist_3 += np.linalg.norm(p - m)
            mf_dist_3 += np.linalg.norm(m - f)
            pf_dist_3 += np.linalg.norm(p - f)
            n_3 += 1

    # FOUR NOTES VOLUME DISTANCE

    maj7_structure = np.array([0, 4, 7, 11])
    m7_structure   = np.array([0, 3, 7, 10])
    M7_structure   = np.array([0, 4, 7, 10])
    m7b5_structure = np.array([0, 3, 6, 10])

    structures = [maj7_structure, m7_structure, M7_structure, m7b5_structure]
    volumes = [1/3, 2/3, 1.0]

    max_base_note = 32                  # of 44 notes. THis way avary chord will have 4 notes 

    n_4 = 0
    pm_dist_4 = 0
    mf_dist_4 = 0
    pf_dist_4 = 0

    for i in range(max_base_note):
        for structure in structures:
            full_structure = (structure+i)

            p_datapont = volumes[0]*np.sum(np.eye(44)[full_structure],axis=0).reshape((1,44))
            m_datapont = volumes[1]*np.sum(np.eye(44)[full_structure],axis=0).reshape((1,44))
            f_datapont = volumes[2]*np.sum(np.eye(44)[full_structure],axis=0).reshape((1,44))

            p, _  = TN_VAE.encoder(p_datapont)
            m, _  = TN_VAE.encoder(m_datapont)
            f, _  = TN_VAE.encoder(f_datapont)

            pm_dist_4 += np.linalg.norm(p - m)
            mf_dist_4 += np.linalg.norm(m - f)
            pf_dist_4 += np.linalg.norm(p - f)
            n_4 += 1


    avg_pm_dist_3 = pm_dist_3/n_3
    avg_mf_dist_3 = mf_dist_3/n_3
    avg_pf_dist_3 = pf_dist_3/n_3

    avg_pm_dist_4 = pm_dist_4/n_4
    avg_mf_dist_4 = mf_dist_4/n_4
    avg_pf_dist_4 = pf_dist_4/n_4

    avg_pm_dist = (pm_dist_3 + pm_dist_4)/(n_3 +n_4)
    avg_mf_dist = (mf_dist_3 + mf_dist_4)/(n_3 +n_4)
    avg_pf_dist = (pf_dist_3 + pf_dist_4)/(n_3 +n_4)

    result_str = "\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}\n".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM']) 
                                                                    
    result_str += "\n\tPM AVG DISTANCE 3 notes: \t{:.5f}".format(avg_pm_dist_3)
    result_str += "\n\tMF AVG DISTANCE 3 notes: \t{:.5f}".format(avg_mf_dist_3)
    result_str += "\n\tPF AVG DISTANCE 3 notes: \t{:.5f}".format(avg_pf_dist_3)

    result_str += "\n"

    result_str += "\n\tPM AVG DISTANCE 4 notes: \t{:.5f}".format(avg_pm_dist_4)
    result_str += "\n\tMF AVG DISTANCE 4 notes: \t{:.5f}".format(avg_mf_dist_4)
    result_str += "\n\tPF AVG DISTANCE 4 notes: \t{:.5f}".format(avg_pf_dist_4)

    result_str += "\n"

    result_str += "\n\tPM AVG DISTANCE  notes: \t{:.5f}".format(avg_pm_dist)
    result_str += "\n\tMF AVG DISTANCE  notes: \t{:.5f}".format(avg_mf_dist)
    result_str += "\n\tPF AVG DISTANCE  notes: \t{:.5f}".format(avg_pf_dist)

    result_str += "\n"

    return result_str
    





                







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
    
                result_txt += vol_avg_distance(RUN_PATH)

    with open(os.path.join(CODE_PATH, 'analysis_results/5_volumes_avg_distance/result.txt'), 'w') as text_file:
        text_file.write(result_txt)






	

    




        
if __name__=="__main__":
    main()