from src.models.pianorollModel import PianoRollModel
from src.lib.generate_chord_latent_pair import main as helper

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree



def model_nearest_neighbour_analysis(run_path, triad_chord, triad_latent, verbose=False):
    
    with open(os.path.join(run_path, 'train_params.txt')) as f:
        H_PARAMS = json.load(f)

    n_3 = np.shape(triad_latent)[0]

    tree = KDTree(triad_latent)

    distances, indices = tree.query(triad_latent, 11)

    n_notes_common_0_array = np.zeros([n_3])
    n_notes_common_1_array = np.zeros([n_3])
    n_notes_common_2_array = np.zeros([n_3])
    n_notes_common_3_array = np.zeros([n_3])
    n_notes_common_4_array = np.zeros([n_3])
    n_notes_common_5_array = np.zeros([n_3])
    n_notes_common_6_array = np.zeros([n_3])
    n_notes_common_7_array = np.zeros([n_3])
    n_notes_common_8_array = np.zeros([n_3])
    n_notes_common_9_array = np.zeros([n_3])

    n_ṿolume_dist_0_array = np.zeros([n_3])
    n_ṿolume_dist_1_array = np.zeros([n_3])
    n_ṿolume_dist_2_array = np.zeros([n_3])
    n_ṿolume_dist_3_array = np.zeros([n_3])
    n_ṿolume_dist_4_array = np.zeros([n_3])
    n_ṿolume_dist_5_array = np.zeros([n_3])
    n_ṿolume_dist_6_array = np.zeros([n_3])
    n_ṿolume_dist_7_array = np.zeros([n_3])
    n_ṿolume_dist_8_array = np.zeros([n_3])
    n_ṿolume_dist_9_array = np.zeros([n_3])


    for i in range(n_3):
        chord_3 = triad_chord[i]

        nearest_0 = triad_chord[indices[i][1]]
        nearest_1 = triad_chord[indices[i][2]]
        nearest_2 = triad_chord[indices[i][3]]
        nearest_3 = triad_chord[indices[i][4]]
        nearest_4 = triad_chord[indices[i][5]]
        nearest_5 = triad_chord[indices[i][6]]
        nearest_6 = triad_chord[indices[i][7]]
        nearest_7 = triad_chord[indices[i][8]]
        nearest_8 = triad_chord[indices[i][9]]
        nearest_9 = triad_chord[indices[i][10]]


        n_notes_common_0_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_0)))
        n_notes_common_1_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_1)))
        n_notes_common_2_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_2)))
        n_notes_common_3_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_3)))
        n_notes_common_4_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_4)))
        n_notes_common_5_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_5)))
        n_notes_common_6_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_6)))
        n_notes_common_7_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_7)))
        n_notes_common_8_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_8)))
        n_notes_common_9_array[i] = np.sum(np.logical_and(np.ceil(chord_3), np.ceil(nearest_9)))


        n_ṿolume_dist_0_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_0)/3)
        n_ṿolume_dist_1_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_1)/3)
        n_ṿolume_dist_2_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_2)/3)
        n_ṿolume_dist_3_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_3)/3)
        n_ṿolume_dist_4_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_4)/3)
        n_ṿolume_dist_5_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_5)/3)
        n_ṿolume_dist_6_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_6)/3)
        n_ṿolume_dist_7_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_7)/3)
        n_ṿolume_dist_8_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_8)/3)
        n_ṿolume_dist_9_array[i] = 3*np.abs(np.sum(chord_3)/3 - np.sum(nearest_9)/3)

    notes_avg_0 = np.sum(n_notes_common_0_array)/n_3
    notes_avg_1 = np.sum(n_notes_common_1_array)/n_3
    notes_avg_2 = np.sum(n_notes_common_2_array)/n_3
    notes_avg_3 = np.sum(n_notes_common_3_array)/n_3
    notes_avg_4 = np.sum(n_notes_common_4_array)/n_3
    notes_avg_5 = np.sum(n_notes_common_5_array)/n_3
    notes_avg_6 = np.sum(n_notes_common_6_array)/n_3
    notes_avg_7 = np.sum(n_notes_common_7_array)/n_3
    notes_avg_8 = np.sum(n_notes_common_8_array)/n_3
    notes_avg_9 = np.sum(n_notes_common_9_array)/n_3

    volumes_avg_0 = np.sum(n_ṿolume_dist_0_array)/n_3
    volumes_avg_1 = np.sum(n_ṿolume_dist_1_array)/n_3
    volumes_avg_2 = np.sum(n_ṿolume_dist_2_array)/n_3
    volumes_avg_3 = np.sum(n_ṿolume_dist_3_array)/n_3
    volumes_avg_4 = np.sum(n_ṿolume_dist_4_array)/n_3
    volumes_avg_5 = np.sum(n_ṿolume_dist_5_array)/n_3
    volumes_avg_6 = np.sum(n_ṿolume_dist_6_array)/n_3
    volumes_avg_7 = np.sum(n_ṿolume_dist_7_array)/n_3
    volumes_avg_8 = np.sum(n_ṿolume_dist_8_array)/n_3
    volumes_avg_9 = np.sum(n_ṿolume_dist_9_array)/n_3

    result_str = "\n\nLATENT DIM: {:d}".format(H_PARAMS['LATENT_DIM']) 

    result_str += "\n"
                                                                    
    result_str += "NOTES_AVG_0:   {:.5f}   ".format(notes_avg_0)
    result_str += "NOTES_AVG_1:   {:.5f}   ".format(notes_avg_1)
    result_str += "NOTES_AVG_2:   {:.5f}   ".format(notes_avg_2)
    result_str += "NOTES_AVG_3:   {:.5f}   ".format(notes_avg_3)
    result_str += "NOTES_AVG_4:   {:.5f}   ".format(notes_avg_4)
    result_str += "NOTES_AVG_5:   {:.5f}   ".format(notes_avg_5)
    result_str += "NOTES_AVG_6:   {:.5f}   ".format(notes_avg_6)
    result_str += "NOTES_AVG_7:   {:.5f}   ".format(notes_avg_7)
    result_str += "NOTES_AVG_8:   {:.5f}   ".format(notes_avg_8)
    result_str += "NOTES_AVG_9:   {:.5f}   ".format(notes_avg_9)

    result_str += "\n"
    
    result_str += "VOLUMES_AVG_0: {:.5f}   ".format(volumes_avg_0)
    result_str += "VOLUMES_AVG_1: {:.5f}   ".format(volumes_avg_1)
    result_str += "VOLUMES_AVG_2: {:.5f}   ".format(volumes_avg_2)
    result_str += "VOLUMES_AVG_3: {:.5f}   ".format(volumes_avg_3)
    result_str += "VOLUMES_AVG_4: {:.5f}   ".format(volumes_avg_4)
    result_str += "VOLUMES_AVG_5: {:.5f}   ".format(volumes_avg_5)
    result_str += "VOLUMES_AVG_6: {:.5f}   ".format(volumes_avg_6)
    result_str += "VOLUMES_AVG_7: {:.5f}   ".format(volumes_avg_7)
    result_str += "VOLUMES_AVG_8: {:.5f}   ".format(volumes_avg_8)
    result_str += "VOLUMES_AVG_9: {:.5f}   ".format(volumes_avg_9)

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
                triad_chord  = np.load(RUN_PATH+'/triad_chord.npy')
                triad_latent = np.load(RUN_PATH+'/triad_latent.npy')
                result_txt += model_nearest_neighbour_analysis(RUN_PATH, triad_chord, triad_latent)

    with open(os.path.join(CODE_PATH, 'analysis_results/5_nearest_neighbour/triad_triad_nearest.txt'), 'w') as text_file:
        text_file.write(result_txt)






	

    




        
if __name__=="__main__":
    helper()
    main()