from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf



def compute_accuracy_precision(data, y_pred_clean, verbose=False):
    TP = np.sum(np.logical_and(data, y_pred_clean))
    TN = np.sum(np.logical_and(np.logical_not(data), np.logical_not(y_pred_clean)))
    FP = np.sum(np.logical_and(np.logical_xor(data, y_pred_clean), y_pred_clean))
    FN = np.sum(np.logical_and(np.logical_xor(data, y_pred_clean), data))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)

    if verbose:
        print(' ')
        print(np.shape(data))
        print(np.shape(y_pred_clean))
        print(TP)
        print(TN)
        print(FP)
        print(FN)
        print(TP+TN+FP+FN)

        print(data[7])
        print(y_pred_clean[7])
        print(np.logical_and(data, y_pred_clean)[7])

    return accuracy, precision

def compute_avg_clean_raw_diff(clean, raw):
    return np.sum(np.abs(clean - raw))/clean.size


def model_get_acccu(run_path, data_path, verbose=False):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    # MODEL
    TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])
    TN_VAE.load_weights(os.path.join(run_path, 'weights/weights.h5'))
    print('MODEL LOADED')

    data = np.load(data_path)
    
    mu, log_var  = TN_VAE.encoder(data)
    y_pred_raw   = TN_VAE.decoder(mu) 
    y_pred_clean = TN_VAE.clean_pianoroll(y_pred_raw)

    full_accuracy, full_precision = compute_accuracy_precision(np.eye(4)[np.array(3*data, np.int16)][:,:,1:], np.eye(4)[np.array(3*y_pred_clean, np.int16)][:,:,1:], verbose)
    note_accuracy, note_precision = compute_accuracy_precision(np.ceil(data), np.ceil(y_pred_clean),verbose)
    avg_clean_raw_diff            = compute_avg_clean_raw_diff(y_pred_clean, y_pred_raw)

    result_str = "\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM']) 
                                                                    
    result_str += "\n\tFULL ACCURACY: {:.5f}   FULL PRECISION: {:.5f}".format(
                                                                    full_accuracy, 
                                                                    full_precision)

    result_str += "\n\tNOTE ACCURACY: {:.5f}   NOTE PRECISION: {:.5f}".format(
                                                                note_accuracy, 
                                                                note_precision)

    result_str += "\n\tAVERAGE CLEAN VS RAW DIFFERENCE: {:.5f}".format(avg_clean_raw_diff)

    return result_str



def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]


    VAL_DATA_PATH          = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/validation_roll.npy')
    FOUR_NOTES_DATA_PATH   = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/4_notes_roll.npy')
    RANDOM_NOTES_DATA_PATH = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/random_notes_roll.npy')
    
    val_result_txt          = ''
    four_notes_result_txt   = ''
    random_notes_result_txt = ''

    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
                val_result_txt          += model_get_acccu(RUN_PATH, VAL_DATA_PATH)
                four_notes_result_txt   += model_get_acccu(RUN_PATH, FOUR_NOTES_DATA_PATH)
                random_notes_result_txt += model_get_acccu(RUN_PATH, RANDOM_NOTES_DATA_PATH, verbose=False)

    with open(os.path.join(CODE_PATH, 'analysis_results/1_reconstruction_accu/val_result.txt'), 'w') as text_file:
        text_file.write(val_result_txt)

    with open(os.path.join(CODE_PATH, 'analysis_results/1_reconstruction_accu/four_notes_result.txt'), 'w') as text_file:
        text_file.write(four_notes_result_txt)
    
    with open(os.path.join(CODE_PATH, 'analysis_results/1_reconstruction_accu/random_notes_result.txt'), 'w') as text_file:
        text_file.write(random_notes_result_txt)


    




        
if __name__=="__main__":
    main()