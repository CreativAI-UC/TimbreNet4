from src.models.pianorollModel import PianoRollModel

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




def explained_variance(dataset, title, CODE_FOLDER, dim, model_path = None):
	if (model_path != None):
		with open(os.path.join(model_path, 'train_params.txt')) as f:
			H_PARAMS = json.load(f)

		# MODEL
		TN_VAE = PianoRollModel(model_latent_dim = H_PARAMS['LATENT_DIM'], hidden_layers=H_PARAMS['HIDDEN_LAYERS'], hidden_layers_dim=H_PARAMS['HIDDEN_LAYERS_DIM'])
		TN_VAE.load_weights(os.path.join(model_path, 'weights/weights.h5'))
		print('MODEL LOADED')

		data = dataset
		mu, log_var  = TN_VAE.encoder(data)

		dataset = mu




	print("ANALYSIS: ")
	text = title + ":\n"
	pca = PCA(n_components=dim)
	principalComponents = pca.fit_transform(dataset)
	cumulate = 0
	cummulate_array = np.zeros(dim)
	for idx, i in enumerate(pca.explained_variance_ratio_):
		cumulate += i
		cummulate_array[idx] = cumulate
		text += "{:02d}:    ".format(idx+1)
		text += "{:.9f}    ".format(i)
		text += "{:.9f}\n".format(cumulate)
		print("{:02d}".format(idx+1), end = ':   ')
		print("{:.9f}".format(i), end = '   ')
		print("{:.9f}".format(cumulate))
	with open(os.path.join(CODE_FOLDER, 'analysis_results/1_PCA/'+title+'.txt'), 'w') as text_file:
		text_file.write(text)

	return cummulate_array



def main():
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	TRAINED_MODELS_PATH = os.path.join(CODE_FOLDER, 'trained_models')
	LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    
	DATASET_PATH = os.path.join(os.path.dirname(CODE_FOLDER),'datasets/numpyDatasets')
	triad_val   = np.load(DATASET_PATH+'/3_notes_roll_val.npy')
	tetrad_val  = np.load(DATASET_PATH+'/4_notes_roll.npy')
	random_val  = np.load(DATASET_PATH+'/3_random_notes_roll.npy')

	x = np.linspace(1, 32, num=32)
	plt.figure(figsize=(8, 8), dpi=200)
	plt.plot(x, x/32, 'k-')
	
	for LATENT_DIM_PATH in LATENT_DIM_PATHS:
		ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
		for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
			RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
			for RUN_PATH in RUN_PATHS:
				RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
				print(LATENT_DIM_PATH[-2:])
				if LATENT_DIM_PATH[-2:] == '32':
					style = '--'
					dim = 32
				elif LATENT_DIM_PATH[-2:] == '16':
					style = '-.'
					dim = 16
				elif LATENT_DIM_PATH[-2:] == '_8':
					style = ':'
					dim = 8
				elif LATENT_DIM_PATH[-2:] == '_4':
					style = '.'
					dim = 4
				plt.plot(x[0:dim], explained_variance(triad_val,  "TRIAD_VALIDATION_LATENT_"+str(dim),  CODE_FOLDER, dim, RUN_PATH), 'b'+style, label='triad validation out ' + LATENT_DIM_PATH)
				plt.plot(x[0:dim], explained_variance(tetrad_val, "TETRAD_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'r'+style, label='tetrad validation out '+ LATENT_DIM_PATH)
				plt.plot(x[0:dim], explained_variance(random_val, "RANDOM_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'g'+style, label='random validation out '+ LATENT_DIM_PATH)



	
	plt.legend()
	plt.grid()

	plt.savefig(os.path.join(CODE_FOLDER, 'analysis_results/1_PCA/latent_PCA_full.png'))
	plt.show()

    




        
if __name__=="__main__":
    main()