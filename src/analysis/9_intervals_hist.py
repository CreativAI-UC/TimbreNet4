import os
import numpy as np
import matplotlib.pyplot as plt

def get_interval_hist(data):
	hist = np.zeros((45))
	n = 0
	for example in data:
		n = n+1
		if (n%10000 == 0):
			print(int(n/10000))
		count = False
		interval = 0
		for element in example:
			if element == 1:
				count = True
				if interval != 0:
					hist[interval] = hist[interval] +1
					interval = 0
			if count:
				interval = interval + 1
	print(hist)
	return hist



def main():
	
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	DATA_FOLDER = os.path.dirname(CODE_FOLDER)+'/datasets/numpyDatasets/'
	'''
	dataset__ = np.load(DATA_FOLDER+'triad_dataset.npy') 
	latent_04 = np.load(DATA_FOLDER+'generated_chords_4.npy')
	latent_08 = np.load(DATA_FOLDER+'generated_chords_8.npy')
	latent_16 = np.load(DATA_FOLDER+'generated_chords_16.npy')
	latent_32 = np.load(DATA_FOLDER+'generated_chords_32.npy')

	dataset__ = np.ceil(dataset__)
	latent_04 = np.ceil(latent_04)
	latent_08 = np.ceil(latent_08)
	latent_16 = np.ceil(latent_16)
	latent_32 = np.ceil(latent_32)

	hist_ds = get_interval_hist(dataset__)
	hist_04 = get_interval_hist(latent_04)
	hist_08 = get_interval_hist(latent_08)
	hist_16 = get_interval_hist(latent_16)
	hist_32 = get_interval_hist(latent_32)

	np.save('/home/agustin/Desktop/tesis_images/hist_ds.npy', hist_ds)
	np.save('/home/agustin/Desktop/tesis_images/hist_04.npy', hist_04)
	np.save('/home/agustin/Desktop/tesis_images/hist_08.npy', hist_08)
	np.save('/home/agustin/Desktop/tesis_images/hist_16.npy', hist_16)
	np.save('/home/agustin/Desktop/tesis_images/hist_32.npy', hist_32)
	'''





	hist_ds = np.load('/home/agustin/Desktop/tesis_images/hist_ds.npy')
	hist_04 = np.load('/home/agustin/Desktop/tesis_images/hist_04.npy')
	hist_08 = np.load('/home/agustin/Desktop/tesis_images/hist_08.npy')
	hist_16 = np.load('/home/agustin/Desktop/tesis_images/hist_16.npy')
	hist_32 = np.load('/home/agustin/Desktop/tesis_images/hist_32.npy')



	hist_ds = hist_ds/np.sum(hist_ds)
	hist_04 = hist_04/np.sum(hist_04)
	hist_08 = hist_08/np.sum(hist_08)
	hist_16 = hist_16/np.sum(hist_16)
	hist_32 = hist_32/np.sum(hist_32)



	plt.figure(figsize=(5, 5), dpi=200)

	bins = np.arange(13)

	plt.bar(bins, hist_ds[0:13], label="dataset__", linewidth=3.4, color='grey', edgecolor='grey')
	plt.bar(bins, hist_04[0:13], label="latent_04", linewidth=3.0, color='none', edgecolor='blue')
	plt.bar(bins, hist_08[0:13], label="latent_08", linewidth=2.6, color='none', edgecolor='red')
	plt.bar(bins, hist_16[0:13], label="latent_16", linewidth=2.2, color='none', edgecolor='green')
	plt.bar(bins, hist_32[0:13], label="latent_32", linewidth=1.8, color='none', edgecolor='orange')
	
	plt.legend(loc='upper right', fontsize=10)
	plt.xticks(np.arange(12))
	plt.grid()
	plt.grid()
	plt.xlabel('Interval (semitones)')
	plt.ylabel('Normalized frequency')


	plt.savefig(os.path.join(CODE_FOLDER, 'analysis_results/9_interval_hist/intervals_histogram.png'), dpi=200)
	plt.show()


				


    




        
if __name__=="__main__":
    main()