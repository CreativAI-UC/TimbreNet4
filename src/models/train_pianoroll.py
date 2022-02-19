from src.models.pianorollModel import PianoRollModel

import os
import json
import pickle
import datetime
import numpy as np
import tensorflow as tf
from distutils.dir_util import copy_tree
from time import sleep

os.environ["CUDA_VISIBLE_DEVICES"]="0"
print('\n\n\n')
print(tf.config.list_physical_devices('GPU'))


def create_run_folder(LATENT_DIM, HIDDEN_LAYERS, HIDEN_LAYERS_DIM):
    '''
    Function that creates a folder where to run the training creating an unique ID
    '''
    CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TIME_CLOCK  = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    RUN_ID = 'ID_'+TIME_CLOCK
    RUN_FOLDER = '{}/trained_models/latent_{}/hidden_{}_dim_{}/{}'.format(CODE_FOLDER, LATENT_DIM, HIDDEN_LAYERS, HIDEN_LAYERS_DIM, RUN_ID)
    
    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
        os.makedirs(os.path.join(RUN_FOLDER, 'weights'))
        os.makedirs(os.path.join(RUN_FOLDER, 'logs'))
        os.makedirs(os.path.join(RUN_FOLDER + '/logs', 'scalars'))

    return RUN_FOLDER, RUN_ID

def train(H_PARAMS):
    '''
    Train script function, creates folder, saves metadata info, uploads weighs
    Perform multile sets of training with digfferent weights fot thre losses
    '''
    for LOSS_FACTOR_ITEM in H_PARAMS['LOSS_FACTOR_SEQUENCE']:
        
        # CREATE RUN FOLDER
        RUN_FOLDER, RUN_ID =create_run_folder(H_PARAMS['LATENT_DIM'], H_PARAMS['HIDDEN_LAYERS'], H_PARAMS['HIDDEN_LAYERS_DIM'])
        print('\n\nRUN_ID: {}'.format(RUN_ID))

        

        # ADD HYPERPARAMETERS
        # If new model, create model
        if H_PARAMS['PARENT_TRAIN_ID'] == None:
            H_PARAMS['MODE'] = 'BUILD'
            H_PARAMS['INITIAL_EPOCH'] = 0
            H_PARAMS['END_EPOCH']     = H_PARAMS['INITIAL_EPOCH'] + LOSS_FACTOR_ITEM['N_EPOCHS']
            H_PARAMS['R_LOSS_FACTOR'] = LOSS_FACTOR_ITEM['R_LOSS_FACTOR']
            H_PARAMS['TRAIN_COMPLETED'] = False

        # Else, training starts from another pre-trained / trained model
        else:
            H_PARAMS['MODE'] = 'LOAD'
            with open(os.path.join(os.path.dirname(RUN_FOLDER), H_PARAMS['PARENT_TRAIN_ID'], 'train_params.txt')) as f:
                OLD_H_PARAMS = json.load(f)
            H_PARAMS['INITIAL_EPOCH'] = OLD_H_PARAMS['END_EPOCH']
            H_PARAMS['END_EPOCH']     = H_PARAMS['INITIAL_EPOCH'] + LOSS_FACTOR_ITEM['N_EPOCHS']
            H_PARAMS['R_LOSS_FACTOR'] = LOSS_FACTOR_ITEM['R_LOSS_FACTOR']
            H_PARAMS['PARENT_TRAIN_ID'] = H_PARAMS['PARENT_TRAIN_ID']
            H_PARAMS['TRAIN_COMPLETED'] = False
            
            copy_tree(os.path.join(os.path.dirname(RUN_FOLDER), H_PARAMS['PARENT_TRAIN_ID']), RUN_FOLDER)
        
        # Save trainig params in human readable text and machine readable info    
        with open(os.path.join(RUN_FOLDER, 'train_params.txt'), 'w') as f:
            f.write(json.dumps(H_PARAMS))

        with open(os.path.join(RUN_FOLDER, 'train_params.pkl'), 'wb') as f:
            pickle.dump(H_PARAMS, f)
       
        # MODEL
        TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])

        if H_PARAMS['MODE'] == 'BUILD':
            TN_VAE.save_model_params(RUN_FOLDER)
            print('MODEL BUILD')
        elif H_PARAMS['MODE'] == 'LOAD':
            TN_VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
            print('MODEL LOADED') 

        # DATASET
        numpy_train_dataset = np.repeat(np.float32(np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(RUN_FOLDER))))), H_PARAMS['TRAIN_DATASET']))),100, axis = 0)
        numpy_val_dataset   = np.float32(np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(RUN_FOLDER))))), H_PARAMS['VAL_DATASET'])))

        train_dataset = tf.data.Dataset.from_tensor_slices((numpy_train_dataset,numpy_train_dataset)).shuffle(H_PARAMS['NUM_TRAIN_EX'], seed=H_PARAMS['SEED'], reshuffle_each_iteration=True).batch(H_PARAMS['BATCH_SIZE'])
        val_dataset  = tf.data.Dataset.from_tensor_slices((numpy_val_dataset, numpy_val_dataset)).shuffle(H_PARAMS['NUM_VAL_EX'], seed=H_PARAMS['SEED'], reshuffle_each_iteration=True).batch(1)




        print('\nMODEL DATASET OK')

        # TRAIN MODEL
        TN_VAE.train_with_generator(
            train_data_flow  = train_dataset,
            val_data_flow    = val_dataset,
            epochs           = LOSS_FACTOR_ITEM['N_EPOCHS'],
            initial_epoch    = H_PARAMS['INITIAL_EPOCH'],
            learning_rate    = H_PARAMS['LEARNING_RATE'],
            r_loss_factor    = LOSS_FACTOR_ITEM['R_LOSS_FACTOR'],
            run_folder       = RUN_FOLDER,
            )

        # CONFIRM TRAIN COMPLETED
        H_PARAMS['TRAIN_COMPLETED'] = True
        with open(os.path.join(RUN_FOLDER, 'train_params.txt'), 'w') as f:
            f.write(json.dumps(H_PARAMS))
        
        H_PARAMS['PARENT_TRAIN_ID'] = RUN_ID
        sleep(3)
        print("RUN COMPLETED")

    print("\nTRAINING COMPLETED")


if __name__=="__main__":
    print("\n\nWelcome to TimbreNet 3 Train Script")

    H_PARAMS = {
        'LATENT_DIM'            : 4,
        'HIDDEN_LAYERS'         : 2,
        'HIDDEN_LAYERS_DIM'     : 16,
        'PARENT_TRAIN_ID'       : 'ID_2022_02_10_19_59_35',#None,
        'TRAIN_DATASET'         : 'datasets/numpyDatasets/3_notes_roll_train.npy',
        'VAL_DATASET'           : 'datasets/numpyDatasets/3_notes_roll_val.npy',
        'NUM_TRAIN_EX'          :  76800,
        'NUM_VAL_EX'            :  192,
        'SEED'                  :   21,

        'LOSS_FACTOR_SEQUENCE'  : [
                                    #{'R_LOSS_FACTOR':3000  ,'N_EPOCHS': 50},
                                    #{'R_LOSS_FACTOR':2000  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR':1000  ,'N_EPOCHS': 150},
                                    {'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    {'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    {'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    

                                    


                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 100},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    

                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},

                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},

                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},

                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},

                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                    #{'R_LOSS_FACTOR': 500  ,'N_EPOCHS': 20},
                                   ],
        'LEARNING_RATE'         : 3e-5,
        'R_LOSS_FACTOR'         : None,
        'META_LOSS_FACTOR'      : None,
        'BATCH_SIZE'            :   10,
    }


    train(H_PARAMS)