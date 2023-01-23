import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats


def main():
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	DATA_FOLDER = os.path.dirname(CODE_FOLDER)+'/datasets/numpyDatasets/'

	dataset__   = np.load(DATA_FOLDER+'triad_dataset.npy') 
	latent_04 = np.load(DATA_FOLDER+'generated_chords_4.npy')
	latent_08 = np.load(DATA_FOLDER+'generated_chords_8.npy')
	latent_16 = np.load(DATA_FOLDER+'generated_chords_16.npy')
	latent_32 = np.load(DATA_FOLDER+'generated_chords_32.npy')

	hist_ds = np.sum(np.ceil(dataset__), axis=1)
	hist_04 = np.sum(np.ceil(latent_04), axis=1)
	hist_08 = np.sum(np.ceil(latent_08), axis=1)
	hist_16 = np.sum(np.ceil(latent_16), axis=1)
	hist_32 = np.sum(np.ceil(latent_32), axis=1)

	print("dataset__", end='\t')
	print(np.mean(hist_ds), end='\t')
	print(np.std(hist_ds), end='\t')
	print(stats.entropy(hist_ds), end='\n')


	print("latent_04", end='\t')
	print(np.mean(hist_04), end='\t')
	print(np.std(hist_04), end='\t')
	print(stats.entropy(hist_04), end='\n')

	print("latent_08", end='\t')
	print(np.mean(hist_08), end='\t')
	print(np.std(hist_08), end='\t')
	print(stats.entropy(hist_08), end='\n')

	print("latent_16", end='\t')
	print(np.mean(hist_16), end='\t')
	print(np.std(hist_16), end='\t')
	print(stats.entropy(hist_16), end='\n')

	print("latent_32", end='\t')
	print(np.mean(hist_32), end='\t')
	print(np.std(hist_32), end='\t')
	print(stats.entropy(hist_32), end='\n')











	x = np.linspace(0,44,45)
	plt.figure(figsize=(5, 5), dpi=200)

	bins_array = np.arange(11) - 0.5

	#plt.hist(hist_04, bins=[0,1,2,3,4,5,6,7,8,9,10], histtype='step', density=True, label="dataset__", linewidth=2.0)
	plt.hist(hist_04, bins=bins_array, histtype='step', linewidth=3.0, density=True, label="latent_04")
	plt.hist(hist_08, bins=bins_array, histtype='step', linewidth=2.6, density=True, label="latent_08")
	plt.hist(hist_16, bins=bins_array, histtype='step', linewidth=2.2, density=True, label="latent_16")
	plt.hist(hist_32, bins=bins_array, histtype='step', linewidth=1.8, density=True, label="latent_32")
	plt.legend()
	plt.xticks(np.arange(10))
	plt.grid()
	plt.xlabel('Number of notes in chord')
	plt.ylabel('Normalized frequency')

	
	plt.savefig(os.path.join(CODE_FOLDER, 'analysis_results/8_num_notes_hist/notes_number_histogram.png'), dpi=200)
	plt.show()


				


    




        
if __name__=="__main__":
    main()