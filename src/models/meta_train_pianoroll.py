import os
from train_pianoroll import train

if __name__=="__main__":
    print("\n\nWelcome to TimbreNet 3 Train Script")
    
    LATENT_DIM = 32
    HIDDEN_LAYERS = 1
    HIDDEN_LAYERS_DIM = 32
    CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    RUN_FOLDER = '{}/trained_models/latent_{}/hidden_{}_dim_{}/'.format(CODE_FOLDER, LATENT_DIM, HIDDEN_LAYERS, HIDDEN_LAYERS_DIM)

    NEWEST_ID = 'ID_2022_00_00_00_00_00'
    n_year = 2021
    n_month = 00
    n_day = 00
    n_hour = 00
    n_minute = 00
    n_second = 00
    for file in os.listdir(RUN_FOLDER):
        d = os.path.join(RUN_FOLDER, file)
        if os.path.isdir(d):
            year = int(file[3:3+4] )
            month = int(file[8:10] )
            day = int(file[11:13] )
            hour = int(file[14:16] )
            minute = int(file[17:19] )
            second = int(file[20:22] )
            if  ((year > n_year)) or ((year== n_year) and (month> n_month)) or((year== n_year) and (month== n_month) and (day> n_day)) or ((year== n_year) and (month== n_month) and (day== n_day) and (hour> n_hour)) or ((year== n_year) and (month== n_month) and (day== n_day) and (hour== n_hour) and (minute> n_minute)) or ((year== n_year) and (month== n_month) and (day== n_day) and (hour== n_hour) and (minute== n_minute) and (second> n_second)):


                NEWEST_ID = file
                n_year = year
                n_month = month
                n_day = day
                n_hour = hour
                n_minute = minute
                n_second = second

    print(NEWEST_ID)


    H_PARAMS = {
        'LATENT_DIM'            : LATENT_DIM,
        'HIDDEN_LAYERS'         : HIDDEN_LAYERS,
        'HIDDEN_LAYERS_DIM'     : HIDDEN_LAYERS_DIM,
        'PARENT_TRAIN_ID'       : NEWEST_ID,
        'TRAIN_DATASET'         : 'datasets/numpyDatasets/triad_train_augmented.npy',
        'VAL_DATASET'           : 'datasets/numpyDatasets/triad_val.npy',
        'NUM_TRAIN_EX'          :  972000,
        'NUM_VAL_EX'            :  45,
        'SEED'                  :   21,

        'LOSS_FACTOR_SEQUENCE'  : [
                                    {'R_LOSS_FACTOR': 50  ,'N_EPOCHS': 50},                               
                                   ],
        'LEARNING_RATE'         : 3e-5,
        'R_LOSS_FACTOR'         : None,
        'META_LOSS_FACTOR'      : None,
        'BATCH_SIZE'            :  243,
    }


    train(H_PARAMS)