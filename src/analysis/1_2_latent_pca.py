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
	cummulate_array = np.zeros(dim+1)
	for idx, i in enumerate(pca.explained_variance_ratio_):
		cumulate += i
		cummulate_array[idx+1] = cumulate
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
	triad_val   = np.load(DATASET_PATH+'/triad_val.npy')
	tetrad_val  = np.load(DATASET_PATH+'/tetrad_val.npy')
	random_val  = np.load(DATASET_PATH+'/random_val.npy')

	x = np.linspace(0, 32, num=33)
	fig, axs = plt.subplots(2, 2)
	fig.set_size_inches(7, 7)
	#fig.suptitle('Latent Space PCA Analysis')



	axs[0,0].set_title('4 Latent Dimensions Model')
	axs[0,1].set_title('8 Latent Dimensions Model')
	axs[1,0].set_title('16 Latent Dimensions Model')
	axs[1,1].set_title('32 Latent Dimensions Model')

	axs[0,0].set(xlabel='Component Number', ylabel='Cumulate Variance')
	axs[0,1].set(xlabel='Component Number', ylabel='Cumulate Variance')
	axs[1,0].set(xlabel='Component Number', ylabel='Cumulate Variance')
	axs[1,1].set(xlabel='Component Number', ylabel='Cumulate Variance')

	
	for LATENT_DIM_PATH in LATENT_DIM_PATHS:
		ARCHITECTURE_PATHS = next(os.walk(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH)))[1]
		for ARCHITECTURE_PATH in ARCHITECTURE_PATHS:
			RUN_PATHS = next(os.walk(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH)))[1]
			for RUN_PATH in RUN_PATHS:
				RUN_PATH = os.path.join(os.path.join(os.path.join(TRAINED_MODELS_PATH, LATENT_DIM_PATH),ARCHITECTURE_PATH), RUN_PATH)
				print(LATENT_DIM_PATH[-2:])
				if LATENT_DIM_PATH[-2:] == '32':
					dim = 32
					axs[1,1].plot(x[0:dim+1], x[0:dim+1]/dim, 'k-')
					axs[1,1].plot(x[0:dim+1], explained_variance(triad_val,  "TRIAD_VALIDATION_LATENT_"+str(dim),  CODE_FOLDER, dim, RUN_PATH), 'y:', label='triad validation latent space')
					axs[1,1].plot(x[0:dim+1], explained_variance(tetrad_val, "TETRAD_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'y-.', label='tetrad validation latent space')
					axs[1,1].plot(x[0:dim+1], explained_variance(random_val, "RANDOM_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'y-', label='random validation latent space')
				elif LATENT_DIM_PATH[-2:] == '16':
					dim = 16
					axs[1,0].plot(x[0:dim+1], x[0:dim+1]/dim, 'k-')
					axs[1,0].plot(x[0:dim+1], explained_variance(triad_val,  "TRIAD_VALIDATION_LATENT_"+str(dim),  CODE_FOLDER, dim, RUN_PATH), 'g:', label='triad validation latent space')
					axs[1,0].plot(x[0:dim+1], explained_variance(tetrad_val, "TETRAD_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'g-.', label='tetrad validation latent space')
					axs[1,0].plot(x[0:dim+1], explained_variance(random_val, "RANDOM_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'g-', label='random validation latent space')
				elif LATENT_DIM_PATH[-2:] == '_8':
					dim = 8
					axs[0,1].plot(x[0:dim+1], x[0:dim+1]/dim, 'k-')
					axs[0,1].plot(x[0:dim+1], explained_variance(triad_val,  "TRIAD_VALIDATION_LATENT_"+str(dim),  CODE_FOLDER, dim, RUN_PATH), 'r:', label='triad validation latent space')
					axs[0,1].plot(x[0:dim+1], explained_variance(tetrad_val, "TETRAD_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'r-.', label='tetrad validation latent space')
					axs[0,1].plot(x[0:dim+1], explained_variance(random_val, "RANDOM_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'r-', label='random validation latent space')
				elif LATENT_DIM_PATH[-2:] == '_4':
					dim = 4
					axs[0,0].plot(x[0:dim+1], x[0:dim+1]/dim, 'k-')
					axs[0,0].plot(x[0:dim+1], explained_variance(triad_val,  "TRIAD_VALIDATION_LATENT_"+str(dim),  CODE_FOLDER, dim, RUN_PATH), 'b:', label='triad validation latent space')
					axs[0,0].plot(x[0:dim+1], explained_variance(tetrad_val, "TETRAD_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'b-.', label='tetrad validation latent space')
					axs[0,0].plot(x[0:dim+1], explained_variance(random_val, "RANDOM_VALIDATION_LATENT_"+str(dim), CODE_FOLDER, dim, RUN_PATH), 'b-', label='random validation latent space')
				



	
	axs[0,0].legend(loc='lower right', fontsize=6)
	axs[0,1].legend(loc='lower right', fontsize=6)
	axs[1,0].legend(loc='lower right', fontsize=6)
	axs[1,1].legend(loc='lower right', fontsize=6)
	axs[0,0].grid()
	axs[0,1].grid()
	axs[1,0].grid()
	axs[1,1].grid()

	box = axs[0,0].get_position()
	box.y0 = box.y0 + 0.03
	box.y1 = box.y1 + 0.03
	axs[0,0].set_position(box)

	box = axs[0,1].get_position()
	box.y0 = box.y0 + 0.03
	box.y1 = box.y1 + 0.03
	axs[0,1].set_position(box)


	box = axs[1,0].get_position()
	box.y0 = box.y0 - 0.03
	box.y1 = box.y1 - 0.03
	axs[1,0].set_position(box)

	box = axs[1,1].get_position()
	box.y0 = box.y0 - 0.03
	box.y1 = box.y1 - 0.03
	axs[1,1].set_position(box)


	fig.savefig(os.path.join(CODE_FOLDER, 'analysis_results/1_PCA/pca_latent.png'), dpi=200)
	plt.show()






        
if __name__=="__main__":
    main()