from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree




	





def model_nearest_neighbour_analysis(run_path, chord_3_notes, chord_3_latent, chord_4_notes, chord_4_latent, verbose=False):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    n_3 = np.shape(chord_3_latent)[0]
    n_4 = np.shape(chord_4_latent)[0]

    tree = KDTree(chord_3_latent)

    distances, indices = tree.query(chord_4_latent, 10)

    n_notes_common_0_array = np.zeros([n_4])
    n_notes_common_1_array = np.zeros([n_4])
    n_notes_common_2_array = np.zeros([n_4])
    n_notes_common_3_array = np.zeros([n_4])
    n_notes_common_4_array = np.zeros([n_4])
    n_notes_common_5_array = np.zeros([n_4])
    n_notes_common_6_array = np.zeros([n_4])
    n_notes_common_7_array = np.zeros([n_4])
    n_notes_common_8_array = np.zeros([n_4])
    n_notes_common_9_array = np.zeros([n_4])

    n_ṿolume_dist_0_array = np.zeros([n_4])
    n_ṿolume_dist_1_array = np.zeros([n_4])
    n_ṿolume_dist_2_array = np.zeros([n_4])
    n_ṿolume_dist_3_array = np.zeros([n_4])
    n_ṿolume_dist_4_array = np.zeros([n_4])
    n_ṿolume_dist_5_array = np.zeros([n_4])
    n_ṿolume_dist_6_array = np.zeros([n_4])
    n_ṿolume_dist_7_array = np.zeros([n_4])
    n_ṿolume_dist_8_array = np.zeros([n_4])
    n_ṿolume_dist_9_array = np.zeros([n_4])


    for i in range(n_4):
        chord_4 = chord_4_notes[i]

        nearest_0 = chord_3_notes[indices[i][0]]
        nearest_1 = chord_3_notes[indices[i][1]]
        nearest_2 = chord_3_notes[indices[i][2]]
        nearest_3 = chord_3_notes[indices[i][3]]
        nearest_4 = chord_3_notes[indices[i][4]]
        nearest_5 = chord_3_notes[indices[i][5]]
        nearest_6 = chord_3_notes[indices[i][6]]
        nearest_7 = chord_3_notes[indices[i][7]]
        nearest_8 = chord_3_notes[indices[i][8]]
        nearest_9 = chord_3_notes[indices[i][9]]


        n_notes_common_0_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_0)))
        n_notes_common_1_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_1)))
        n_notes_common_2_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_2)))
        n_notes_common_3_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_3)))
        n_notes_common_4_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_4)))
        n_notes_common_5_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_5)))
        n_notes_common_6_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_6)))
        n_notes_common_7_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_7)))
        n_notes_common_8_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_8)))
        n_notes_common_9_array[i] = np.sum(np.logical_and(np.ceil(chord_4), np.ceil(nearest_9)))


        n_ṿolume_dist_0_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_0)/3)
        n_ṿolume_dist_1_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_1)/3)
        n_ṿolume_dist_2_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_2)/3)
        n_ṿolume_dist_3_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_3)/3)
        n_ṿolume_dist_4_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_4)/3)
        n_ṿolume_dist_5_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_5)/3)
        n_ṿolume_dist_6_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_6)/3)
        n_ṿolume_dist_7_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_7)/3)
        n_ṿolume_dist_8_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_8)/3)
        n_ṿolume_dist_9_array[i] = 3*np.abs(np.sum(chord_4)/4 - np.sum(nearest_9)/3)

    notes_avg_0 = np.sum(n_notes_common_0_array)/n_4
    notes_avg_1 = np.sum(n_notes_common_1_array)/n_4
    notes_avg_2 = np.sum(n_notes_common_2_array)/n_4
    notes_avg_3 = np.sum(n_notes_common_3_array)/n_4
    notes_avg_4 = np.sum(n_notes_common_4_array)/n_4
    notes_avg_5 = np.sum(n_notes_common_5_array)/n_4
    notes_avg_6 = np.sum(n_notes_common_6_array)/n_4
    notes_avg_7 = np.sum(n_notes_common_7_array)/n_4
    notes_avg_8 = np.sum(n_notes_common_8_array)/n_4
    notes_avg_9 = np.sum(n_notes_common_9_array)/n_4

    volumes_avg_0 = np.sum(n_ṿolume_dist_0_array)/n_4
    volumes_avg_1 = np.sum(n_ṿolume_dist_1_array)/n_4
    volumes_avg_2 = np.sum(n_ṿolume_dist_2_array)/n_4
    volumes_avg_3 = np.sum(n_ṿolume_dist_3_array)/n_4
    volumes_avg_4 = np.sum(n_ṿolume_dist_4_array)/n_4
    volumes_avg_5 = np.sum(n_ṿolume_dist_5_array)/n_4
    volumes_avg_6 = np.sum(n_ṿolume_dist_6_array)/n_4
    volumes_avg_7 = np.sum(n_ṿolume_dist_7_array)/n_4
    volumes_avg_8 = np.sum(n_ṿolume_dist_8_array)/n_4
    volumes_avg_9 = np.sum(n_ṿolume_dist_9_array)/n_4

    result_str = "\n\nLATENT DIM: {:d}   HIDDEN LAYERS: {:d}   HIDDEN LAYERS_DIM: {:d}\n".format(
                                                                                    H_PARAMS['LATENT_DIM'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS'], 
                                                                                    H_PARAMS['HIDDEN_LAYERS_DIM']) 
                                                                    
    result_str += "\tNOTES_AVG_0: \t{:.5f}".format(notes_avg_0)
    result_str += "\tNOTES_AVG_1: \t{:.5f}".format(notes_avg_1)
    result_str += "\tNOTES_AVG_2: \t{:.5f}".format(notes_avg_2)
    result_str += "\tNOTES_AVG_3: \t{:.5f}".format(notes_avg_3)
    result_str += "\tNOTES_AVG_4: \t{:.5f}".format(notes_avg_4)
    result_str += "\tNOTES_AVG_5: \t{:.5f}".format(notes_avg_5)
    result_str += "\tNOTES_AVG_6: \t{:.5f}".format(notes_avg_6)
    result_str += "\tNOTES_AVG_7: \t{:.5f}".format(notes_avg_7)
    result_str += "\tNOTES_AVG_8: \t{:.5f}".format(notes_avg_8)
    result_str += "\tNOTES_AVG_9: \t{:.5f}".format(notes_avg_9)

    result_str += "\n"
    
    result_str += "\tVOLUMES_AVG_0: \t{:.5f}".format(volumes_avg_0)
    result_str += "\tVOLUMES_AVG_1: \t{:.5f}".format(volumes_avg_1)
    result_str += "\tVOLUMES_AVG_2: \t{:.5f}".format(volumes_avg_2)
    result_str += "\tVOLUMES_AVG_3: \t{:.5f}".format(volumes_avg_3)
    result_str += "\tVOLUMES_AVG_4: \t{:.5f}".format(volumes_avg_4)
    result_str += "\tVOLUMES_AVG_5: \t{:.5f}".format(volumes_avg_5)
    result_str += "\tVOLUMES_AVG_6: \t{:.5f}".format(volumes_avg_6)
    result_str += "\tVOLUMES_AVG_7: \t{:.5f}".format(volumes_avg_7)
    result_str += "\tVOLUMES_AVG_8: \t{:.5f}".format(volumes_avg_8)
    result_str += "\tVOLUMES_AVG_9: \t{:.5f}".format(volumes_avg_9)

    return result_str



def main():
    CODE_PATH           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    TRAINED_MODELS_PATH = os.path.join(CODE_PATH, 'trained_models')
    LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    THREE_NOTES_DATA_PATH   = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/3_notes_roll.npy')
    FOUR_NOTES_DATA_PATH    = os.path.join(os.path.dirname(CODE_PATH),'datasets/numpyDatasets/4_notes_roll.npy')

    data_3_notes = np.load(THREE_NOTES_DATA_PATH)
    data_4_notes = np.load(FOUR_NOTES_DATA_PATH)

    result_txt = ''


    for LATENT_DIM_PATH in LATENT_DIM_PATHS:
        ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
        for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
            RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
            for RUN_PATH in RUN_PATHS:
                RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
                chord_3_notes  = np.load(RUN_PATH+'/3_notes_chords.npy')
                chord_3_latent = np.load(RUN_PATH+'/3_notes_latent.npy')
                chord_4_notes  = np.load(RUN_PATH+'/4_notes_chords.npy')
                chord_4_latent = np.load(RUN_PATH+'/4_notes_latent.npy')
                result_txt += model_nearest_neighbour_analysis(RUN_PATH, chord_3_notes, chord_3_latent, chord_4_notes, chord_4_latent)

    with open(os.path.join(CODE_PATH, 'analysis_results/4_4_notes_nearest_neighbour/result.txt'), 'w') as text_file:
        text_file.write(result_txt)






	

    




        
if __name__=="__main__":
    main()