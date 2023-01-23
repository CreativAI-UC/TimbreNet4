import os
import numpy as np
import matplotlib.pyplot as plt


def main():
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	DATA_FOLDER = os.path.dirname(CODE_FOLDER)+'/datasets/numpyDatasets/'

	dataset   = np.load(DATA_FOLDER+'triad_dataset.npy') 
	latent_4  = np.load(DATA_FOLDER+'generated_chords_4.npy')
	latent_8  = np.load(DATA_FOLDER+'generated_chords_8.npy')
	latent_16 = np.load(DATA_FOLDER+'generated_chords_16.npy')
	latent_32 = np.load(DATA_FOLDER+'generated_chords_32.npy')

	hist_d  = np.sum(np.ceil(dataset),   axis=0)/np.sum(np.ceil(dataset))
	hist_4  = np.sum(np.ceil(latent_4),  axis=0)/np.sum(np.ceil(latent_4))
	hist_8  = np.sum(np.ceil(latent_8),  axis=0)/np.sum(np.ceil(latent_8))
	hist_16 = np.sum(np.ceil(latent_16), axis=0)/np.sum(np.ceil(latent_16))
	hist_32 = np.sum(np.ceil(latent_32), axis=0)/np.sum(np.ceil(latent_32))

	x = np.linspace(0,44,45)
	plt.figure(figsize=(6, 6), dpi=200)
	
	plt.plot(x, hist_d,  label="dataset", linewidth=7.0)
	plt.plot(x, hist_4,  label="latent_4")
	plt.plot(x, hist_8,  label="latent_8")
	plt.plot(x, hist_16, label="latent_16")
	plt.plot(x, hist_32, label="latent_32")


	plt.legend(loc='lower center', fontsize=10)
	plt.grid()
	plt.xlabel('Note number')
	plt.ylabel('Normalized frequency')

	

	plt.savefig(os.path.join(CODE_FOLDER, 'analysis_results/7_notes_histogram/notes_histogram.png'), dpi=200)
	plt.show()


				


    




        
if __name__=="__main__":
    main()