from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf

from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA



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

    
    


    # NAIVE FORM

    p_latent_std = np.std(p_latent, axis=0)
    m_latent_std = np.std(m_latent, axis=0)
    f_latent_std = np.std(f_latent, axis=0)

    p_latent_mu = np.mean(p_latent, axis=0)
    m_latent_mu = np.mean(m_latent, axis=0)
    f_latent_mu = np.mean(f_latent, axis=0)

    latent_mu = np.stack((p_latent_mu, m_latent_mu, f_latent_mu))
    latent_std = np.std(latent_mu, axis=0)

    naive_result_txt = str("\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}\n".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM']))

    naive_result_txt += str("\np: ")
    naive_result_txt += str((-p_latent_std).argsort()[-4:][::-1])
    naive_result_txt += str("\nm: ")
    naive_result_txt += str((-m_latent_std).argsort()[-4:][::-1])
    naive_result_txt += str("\nf: ")
    naive_result_txt += str((-f_latent_std).argsort()[-4:][::-1])
    naive_result_txt += str("\n\n   ")
    naive_result_txt += str(latent_std.argsort()[-4:][::-1])


    naive_result_txt += str(" ")








    # PCA FORM
    pca = PCA(1)#n_components=H_PARAMS['LATENT_DIM'])
    pca.fit(latent_mu)
    vol_dir = pca.components_
    vol_var = pca.explained_variance_ratio_

    p_pca = PCA(n_components=H_PARAMS['LATENT_DIM'])
    p_pca.fit(p_latent)
    p_vol_dir = p_pca.components_
    p_vol_var = p_pca.explained_variance_ratio_

    m_pca = PCA(n_components=H_PARAMS['LATENT_DIM'])
    m_pca.fit(m_latent)
    m_vol_dir = m_pca.components_
    m_vol_var = m_pca.explained_variance_ratio_

    f_pca = PCA(n_components=H_PARAMS['LATENT_DIM'])
    f_pca.fit(f_latent)
    f_vol_dir = f_pca.components_
    f_vol_var = f_pca.explained_variance_ratio_




    def how_parallel(a1, a2):
        return np.abs(np.dot(a1/np.linalg.norm(a1), a2/np.linalg.norm(a2)))


    pca_result_txt = str("\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}\n".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM']))

    i = H_PARAMS['LATENT_DIM'] -1

    pca_result_txt += str('\n')
    pca_result_txt += str(how_parallel(vol_dir, p_vol_dir[i]))
    pca_result_txt += str('\n')
    pca_result_txt += str(how_parallel(vol_dir, m_vol_dir[i]))
    pca_result_txt += str('\n')
    pca_result_txt += str(how_parallel(vol_dir, f_vol_dir[i]))
    pca_result_txt += str('\n')
    pca_result_txt += str('\n')
    pca_result_txt += str(how_parallel(f_vol_dir[i], m_vol_dir[i]))
    pca_result_txt += str('\n')
    pca_result_txt += str(how_parallel(m_vol_dir[i], p_vol_dir[i]))
    pca_result_txt += str('\n')
    pca_result_txt += str(how_parallel(f_vol_dir[i], p_vol_dir[i]))
    pca_result_txt += str('\n')
    pca_result_txt += str('\n')
    pca_result_txt += str(vol_dir)
    pca_result_txt += str('\n')
    pca_result_txt += str(vol_var)
    pca_result_txt += str('\n')

    if H_PARAMS['LATENT_DIM'] == 3:

        bias = np.random.normal(size=(1,H_PARAMS['LATENT_DIM']))
        n = 20

        for i in range(n):
            j = 6*i/n-3
            roll = TN_VAE.clean_pianoroll(TN_VAE.decoder(j*vol_dir+bias))
            for k in range(44):
                print('{:.1f}'.format(roll[0,k]), end='   ')
            print('\n', end='')



        vol_dir_2 = (f_vol_dir + m_vol_dir + p_vol_dir)/3

        print(' ')
        print(' ')

        bias = np.random.normal(size=(1,H_PARAMS['LATENT_DIM']))
        n = 20

        for m in range(4):
            for i in range(n):
                j = 6*i/n-3
                roll = TN_VAE.clean_pianoroll(TN_VAE.decoder(j*vol_dir_2+bias))
                for k in range(44):
                    print('{:.1f}'.format(roll[m,k]), end='   ')
                print('\n', end='')
            print(' ')
            print(' ')


        
        print(' ')
        print(' ')
        print('aaa')
        print(' ')
        print(' ')

        n = 20

        for m in range(4):
            for i in range(n):
                j = 6*i/n-3
                roll = TN_VAE.clean_pianoroll(TN_VAE.decoder(j*p_vol_dir+p_latent_mu))
                for k in range(44):
                    print('{:.1f}'.format(roll[m,k]), end=' ')
                print('\n', end='')
            print(' ')
            print(' ')



    if H_PARAMS['LATENT_DIM'] == 8:

        for i in range(50):
            bias = np.random.normal(size=(1,H_PARAMS['LATENT_DIM']))
            roll = 10*TN_VAE.clean_pianoroll(TN_VAE.decoder(bias))
            for k in range(44):
                print('{:2d}'.format(int(roll[0,k])), end=' ')
            print('\n\n', end='')


        a = b
    
    
    return naive_result_txt, pca_result_txt, vol_dir, vol_var



                







def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    vol_naive_result_txt = ''
    vol_pca_result_txt  = ''


    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
    
                vol_naive_result_txt_prov, vol_pca_result_txt_prov, vol_dir, vol_var = vol_dim_identification(RUN_PATH)
                vol_naive_result_txt += vol_naive_result_txt_prov
                vol_pca_result_txt   += vol_pca_result_txt_prov

    with open(os.path.join(CODE_PATH, 'analysis_results/6_1_volume_dim_identification/vol_naive_result.txt'), 'w') as text_file:
        text_file.write(vol_naive_result_txt)

    with open(os.path.join(CODE_PATH, 'analysis_results/6_1_volume_dim_identification/vol_pca_result.txt'), 'w') as text_file:
        text_file.write(vol_pca_result_txt)

    print(vol_dir)
    print(vol_var)





	

    




        
if __name__=="__main__":
    main()