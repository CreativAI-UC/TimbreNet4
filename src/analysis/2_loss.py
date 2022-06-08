from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf



def model_get_losses(run_path, data_path):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    # MODEL
    TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])
    TN_VAE.load_weights(os.path.join(run_path, 'weights/weights.h5'))
    print('MODEL LOADED')

    data = np.load(data_path)
    data_flow  = tf.data.Dataset.from_tensor_slices((tf.cast(data, tf.float32), tf.cast(data, tf.float32))).shuffle(H_PARAMS['NUM_VAL_EX'], seed=H_PARAMS['SEED'], reshuffle_each_iteration=True).batch(1)


    TN_VAE.r_loss_factor = 50
    
    val_total_loss  = tf.keras.metrics.Mean()
    val_r_loss      = tf.keras.metrics.Mean()
    val_kl_loss     = tf.keras.metrics.Mean()

    for batch_number, val_data_point in enumerate(data_flow):
        total_loss, r_loss, kl_loss = TN_VAE.compute_loss(val_data_point)
        val_total_loss(total_loss)
        val_r_loss(r_loss)
        val_kl_loss(kl_loss)

    


    result_str = "\n\nLATENT DIM: {:d}   RECONSTRUCTION LOSS: {:.5f}   KL LOSS: {:.5f}".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    val_r_loss.result().numpy(), 
                                                                                    val_kl_loss.result().numpy()) 
    return result_str



def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]


    VAL_DATA_PATH          = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/triad_val.npy')
    FOUR_NOTES_DATA_PATH   = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/tetrad_val.npy')
    RANDOM_NOTES_DATA_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/random_val.npy')
    
    val_result_txt          = ''
    four_notes_result_txt   = ''
    random_notes_result_txt = ''

    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
                verbose = False
                val_result_txt          += model_get_losses(RUN_PATH, VAL_DATA_PATH)
                four_notes_result_txt   += model_get_losses(RUN_PATH, FOUR_NOTES_DATA_PATH)
                random_notes_result_txt += model_get_losses(RUN_PATH, RANDOM_NOTES_DATA_PATH)

    with open(os.path.join(CODE_PATH, 'analysis_results/2_losses/triad_result.txt'), 'w') as text_file:
        text_file.write(val_result_txt)

    with open(os.path.join(CODE_PATH, 'analysis_results/2_losses/tetrad_result.txt'), 'w') as text_file:
        text_file.write(four_notes_result_txt)
    
    with open(os.path.join(CODE_PATH, 'analysis_results/2_losses/random_result.txt'), 'w') as text_file:
        text_file.write(random_notes_result_txt)


    




        
if __name__=="__main__":
    main()