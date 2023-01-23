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
		y_pred_raw   = TN_VAE.decoder(mu) 

		dataset = y_pred_raw




	print("ANALYSIS: ")
	text = title + ":\n"
	pca = PCA(n_components=dim)
	principalComponents = pca.fit_transform(dataset)
	cumulate = 0
	cummulate_array = np.zeros(dim+1)
	for idx, i in enumerate(pca.explained_variance_ratio_):
		cumulate += i
		cummulate_array[idx+1] = cumulate
		text += "{:02d}:    ".format(idx+1)
		text += "{:.6f}    ".format(i)
		text += "{:.6f}\n".format(cumulate)
		print("{:02d}".format(idx+1), end = ':   ')
		print("{:.6f}".format(i), end = '   ')
		print("{:.6f}".format(cumulate))
	with open(os.path.join(CODE_FOLDER, 'analysis_results/1_PCA/'+title+'.txt'), 'w') as text_file:
		text_file.write(text)

	return cummulate_array



def main():
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	TRAINED_MODELS_PATH = os.path.join(CODE_FOLDER, 'trained_models')
	LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    
	DATASET_PATH = os.path.join(os.path.dirname(CODE_FOLDER),'datasets/numpyDatasets')
	triad_val   = np.load(DATASET_PATH+'/triad_val.npy')
	tetrad_val  = np.load(DATASET_PATH+'/tetrad_val.npy')
	random_val  = np.load(DATASET_PATH+'/random_val.npy')

	triad_val_in_array  = explained_variance(triad_val,  "TRIAD_VALIDATION_IN", CODE_FOLDER, 45)
	tetrad_val_in_array = explained_variance(tetrad_val, "TETRAD_VALIDATION_IN", CODE_FOLDER, 45)
	random_val_in_array = explained_variance(random_val, "RANDOM_VALIDATION_IN", CODE_FOLDER, 45)


	x = np.linspace(0, 45, num=46)
	plt.figure(figsize=(4.2, 4.2), dpi=200)
	plt.plot(x, x/45, 'k-')
	
	
	'''
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
				plt.plot(x, explained_variance(triad_val,  "TRIAD_VALIDATION_OUT_"+str(dim),  CODE_FOLDER, 45, RUN_PATH), 'b'+style, label='triad validation out ' + LATENT_DIM_PATH)
				plt.plot(x, explained_variance(tetrad_val, "TETRAD_VALIDATION_OUT_"+str(dim), CODE_FOLDER, 45, RUN_PATH), 'r'+style, label='tetrad validation out '+ LATENT_DIM_PATH)
				plt.plot(x, explained_variance(random_val, "RANDOM_VALIDATION_OUT_"+str(dim), CODE_FOLDER, 45, RUN_PATH), 'g'+style, label='random validation out '+ LATENT_DIM_PATH)
	'''	

	plt.plot(x, triad_val_in_array,  'b-', label='Triad Validation Dataset')
	plt.plot(x, tetrad_val_in_array, 'r-', label='Tetrad Validation Dataset')
	plt.plot(x, random_val_in_array, 'g-', label='Random Validation Dataset')

	
	plt.legend(loc='lower right', fontsize=10)
	plt.grid()
	plt.xlabel('Component Number')
	plt.ylabel('Cumulate Variance')
	#plt.title('Validation Datasets PCA Analysis')

	plt.savefig(os.path.join(CODE_FOLDER, 'analysis_results/1_PCA/pca_validation_datasets.png'))
	plt.show()


def main2():
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	TRAINED_MODELS_PATH = os.path.join(CODE_FOLDER, 'trained_models')
	LATENT_DIM_PATHS    = next(os.walk(TRAINED_MODELS_PATH))[1]

    
	DATASET_PATH = os.path.join(os.path.dirname(CODE_FOLDER),'datasets/numpyDatasets')
	triad_val   = np.load(DATASET_PATH+'/triad_val.npy')
	tetrad_val  = np.load(DATASET_PATH+'/tetrad_val.npy')
	random_val  = np.load(DATASET_PATH+'/random_val.npy')

	triad_val_in_array  = explained_variance(triad_val,  "TRIAD_VALIDATION_IN", CODE_FOLDER, 45)
	tetrad_val_in_array = explained_variance(tetrad_val, "TETRAD_VALIDATION_IN", CODE_FOLDER, 45)
	random_val_in_array = explained_variance(random_val, "RANDOM_VALIDATION_IN", CODE_FOLDER, 45)


	x = np.linspace(0, 45, num=46)


	
	#plt.figure(figsize=(8, 8), dpi=200)
	fig, axs = plt.subplots(2, 2)
	fig.set_size_inches(8, 8)

	

	#fig.suptitle('Validation Datasets Reconstruction PCA Analysis')


	axs[0,0].plot(x, x/45, 'k-')
	axs[0,1].plot(x, x/45, 'k-')
	axs[1,0].plot(x, x/45, 'k-')

	axs[0,0].set_title('Triad Validation Dataset')
	axs[0,1].set_title('Tetrad Validation Dataset')
	axs[1,0].set_title('Random Validation Dataset')

	
	
	axs[0,0].set(xlabel='Component Number', ylabel='Cumulate Variance')
	axs[0,1].set(xlabel='Component Number', ylabel='Cumulate Variance')
	axs[1,0].set(xlabel='Component Number', ylabel='Cumulate Variance')
	
	
	
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
				axs[0,0].plot(x, explained_variance(triad_val,  "TRIAD_VALIDATION_OUT_"+str(dim),  CODE_FOLDER, 45, RUN_PATH), 'b'+style, label=' Reconstructed Dataset ' + str(dim)+' Latent Dimensions Model')
				axs[0,1].plot(x, explained_variance(tetrad_val, "TETRAD_VALIDATION_OUT_"+str(dim), CODE_FOLDER, 45, RUN_PATH), 'r'+style, label=' Reconstructed Dataset ' + str(dim)+' Latent Dimensions Model')
				axs[1,0].plot(x, explained_variance(random_val, "RANDOM_VALIDATION_OUT_"+str(dim), CODE_FOLDER, 45, RUN_PATH), 'g'+style, label=' Reconstructed Dataset ' + str(dim)+' Latent Dimensions Model')
	

	axs[0,0].plot(x, triad_val_in_array,  'b-', label='Original Dataset')
	axs[0,1].plot(x, tetrad_val_in_array, 'r-', label='Original Dataset')
	axs[1,0].plot(x, random_val_in_array, 'g-', label='Original Dataset')

	
	axs[0,0].legend(loc='lower right', fontsize=6)
	axs[0,1].legend(loc='lower right', fontsize=6)
	axs[1,0].legend(loc='lower right', fontsize=6)
	axs[0,0].grid()
	axs[0,1].grid()
	axs[1,0].grid()
	fig.delaxes(axs[1,1])

	box = axs[1,0].get_position()
	box.x0 = box.x0 + 0.2
	box.x1 = box.x1 + 0.2
	box.y0 = box.y0 - 0.02
	box.y1 = box.y1 - 0.02
	axs[1,0].set_position(box)


	fig.savefig(os.path.join(CODE_FOLDER, 'analysis_results/1_PCA/pca_validation_datasets_reconstruction.png'), dpi=200)
	plt.show()

    




        
if __name__=="__main__":
	main()
	main2()