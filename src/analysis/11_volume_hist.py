import os
import numpy as np
import matplotlib.pyplot as plt


def get_volume_statistics(data):
	data = np.floor(3*data+0.5).astype(int)
	data = np.stack((data,data,data), axis=-1)
	for i in range(3):
		for example in data:
			for element in example:
				if element[i] != i+1:
					element[i] = 0
				else:
					element[i] = 1
	data = np.sum(data, axis = 1)
	p_note_being_volume = np.sum(data,axis=0)/np.sum(data)
	data = np.ceil(data/3)
	n_vol = np.sum(data, axis = 1)
	p_vol_in_chord = np.sum(data,axis=0)/np.shape(data)[0]


	n_vol = np.stack((n_vol,n_vol,n_vol,n_vol), axis=-1)
	for i in range(4):
		for element in n_vol:
			if element[i] != i:
				element[i] = 0
			else:
				element[i] = 1
	n_vol = np.sum(n_vol, axis = 0)
	p_num_of_vol = n_vol/np.sum(n_vol)


	print('\n-------\n')
	print('p_note_being_volume')
	print(p_note_being_volume)

	print('\n-------\n')
	print('p_vol_in_chord')
	print(p_vol_in_chord)

	print('\n-------\n')
	print('p_num_of_vol')
	print(p_num_of_vol)

	return p_note_being_volume, p_vol_in_chord, p_num_of_vol


def main():
	
	CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	DATA_FOLDER = os.path.dirname(CODE_FOLDER)+'/datasets/numpyDatasets/'

	'''
	dataset__   = np.load(DATA_FOLDER+'triad_dataset.npy') 
	latent_04 = np.load(DATA_FOLDER+'generated_chords_4.npy')
	latent_08 = np.load(DATA_FOLDER+'generated_chords_8.npy')
	latent_16 = np.load(DATA_FOLDER+'generated_chords_16.npy')
	latent_32 = np.load(DATA_FOLDER+'generated_chords_32.npy')

	p_note_being_volume_ds, p_vol_in_chord_ds, p_num_of_vol_ds = get_volume_statistics(dataset__)
	p_note_being_volume_04, p_vol_in_chord_04, p_num_of_vol_04 = get_volume_statistics(latent_04)
	p_note_being_volume_08, p_vol_in_chord_08, p_num_of_vol_08 = get_volume_statistics(latent_08)
	p_note_being_volume_16, p_vol_in_chord_16, p_num_of_vol_16 = get_volume_statistics(latent_16)
	p_note_being_volume_32, p_vol_in_chord_32, p_num_of_vol_32 = get_volume_statistics(latent_32)

	np.save('/home/agustin/Desktop/tesis_images/p_note_being_volume_ds.npy', p_note_being_volume_ds)
	np.save('/home/agustin/Desktop/tesis_images/p_note_being_volume_04.npy', p_note_being_volume_04)
	np.save('/home/agustin/Desktop/tesis_images/p_note_being_volume_08.npy', p_note_being_volume_08)
	np.save('/home/agustin/Desktop/tesis_images/p_note_being_volume_16.npy', p_note_being_volume_16)
	np.save('/home/agustin/Desktop/tesis_images/p_note_being_volume_32.npy', p_note_being_volume_32)

	np.save('/home/agustin/Desktop/tesis_images/p_vol_in_chord_ds.npy', p_vol_in_chord_ds)
	np.save('/home/agustin/Desktop/tesis_images/p_vol_in_chord_04.npy', p_vol_in_chord_04)
	np.save('/home/agustin/Desktop/tesis_images/p_vol_in_chord_08.npy', p_vol_in_chord_08)
	np.save('/home/agustin/Desktop/tesis_images/p_vol_in_chord_16.npy', p_vol_in_chord_16)
	np.save('/home/agustin/Desktop/tesis_images/p_vol_in_chord_32.npy', p_vol_in_chord_32)

	np.save('/home/agustin/Desktop/tesis_images/p_num_of_vol_ds.npy', p_num_of_vol_ds)
	np.save('/home/agustin/Desktop/tesis_images/p_num_of_vol_04.npy', p_num_of_vol_04)
	np.save('/home/agustin/Desktop/tesis_images/p_num_of_vol_08.npy', p_num_of_vol_08)
	np.save('/home/agustin/Desktop/tesis_images/p_num_of_vol_16.npy', p_num_of_vol_16)
	np.save('/home/agustin/Desktop/tesis_images/p_num_of_vol_32.npy', p_num_of_vol_32)
	'''







	p_note_being_volume_ds = np.load('/home/agustin/Desktop/tesis_images/p_note_being_volume_ds.npy')
	p_note_being_volume_04 = np.load('/home/agustin/Desktop/tesis_images/p_note_being_volume_04.npy')
	p_note_being_volume_08 = np.load('/home/agustin/Desktop/tesis_images/p_note_being_volume_08.npy')
	p_note_being_volume_16 = np.load('/home/agustin/Desktop/tesis_images/p_note_being_volume_16.npy')
	p_note_being_volume_32 = np.load('/home/agustin/Desktop/tesis_images/p_note_being_volume_32.npy')

	p_vol_in_chord_ds = np.load('/home/agustin/Desktop/tesis_images/p_vol_in_chord_ds.npy')
	p_vol_in_chord_04 = np.load('/home/agustin/Desktop/tesis_images/p_vol_in_chord_04.npy')
	p_vol_in_chord_08 = np.load('/home/agustin/Desktop/tesis_images/p_vol_in_chord_08.npy')
	p_vol_in_chord_16 = np.load('/home/agustin/Desktop/tesis_images/p_vol_in_chord_16.npy')
	p_vol_in_chord_32 = np.load('/home/agustin/Desktop/tesis_images/p_vol_in_chord_32.npy')

	p_num_of_vol_ds = np.load('/home/agustin/Desktop/tesis_images/p_num_of_vol_ds.npy')
	p_num_of_vol_04 = np.load('/home/agustin/Desktop/tesis_images/p_num_of_vol_04.npy')
	p_num_of_vol_08 = np.load('/home/agustin/Desktop/tesis_images/p_num_of_vol_08.npy')
	p_num_of_vol_16 = np.load('/home/agustin/Desktop/tesis_images/p_num_of_vol_16.npy')
	p_num_of_vol_32 = np.load('/home/agustin/Desktop/tesis_images/p_num_of_vol_32.npy')



	

	plt.figure(figsize=(8, 8), dpi=200)

	x = np.linspace(1,3,3)
	ax1 = plt.subplot(2,2,1)
	ax1.set_xticks(x)
	plt.title("Proportion of\n a note being (x)\n volume")
	plt.bar(x, p_note_being_volume_ds, linewidth=3.4, label="dataset", color='grey', edgecolor='grey')
	plt.bar(x, p_note_being_volume_04, linewidth=3.0, label="latent_04", color='none', edgecolor='blue')
	plt.bar(x, p_note_being_volume_08, linewidth=2.6, label="latent_08", color='none', edgecolor='red')
	plt.bar(x, p_note_being_volume_16, linewidth=2.2, label="latent_16", color='none', edgecolor='green')
	plt.bar(x, p_note_being_volume_32, linewidth=1.8, label="latent_32", color='none', edgecolor='orange')
	ax1.set_xticklabels(['p', 'm', 'f'])
	plt.legend()

	x = np.linspace(1,3,3)
	ax2 = plt.subplot(2,2,2)
	ax2.set_xticks(x)
	plt.title("Proportion of\n chord having a note with (x)\n volume")
	plt.bar(x, p_vol_in_chord_ds, linewidth=3.4, label="dataset", color='grey', edgecolor='grey')
	plt.bar(x, p_vol_in_chord_04, linewidth=3.0, label="latent_04", color='none', edgecolor='blue')
	plt.bar(x, p_vol_in_chord_08, linewidth=2.6, label="latent_08", color='none', edgecolor='red')
	plt.bar(x, p_vol_in_chord_16, linewidth=2.2, label="latent_16", color='none', edgecolor='green')
	plt.bar(x, p_vol_in_chord_32, linewidth=1.8, label="latent_32", color='none', edgecolor='orange')
	ax2.set_xticklabels(['p', 'm', 'f'])
	plt.legend()

	x = np.linspace(0,3,4)
	ax3 = plt.subplot(2,2,3)
	ax3.set_xticks(x)
	plt.title("Proportion of\n a chord having (x)\n different volumes")
	plt.bar(x, p_num_of_vol_ds, linewidth=3.4, label="dataset", color='grey', edgecolor='grey')
	plt.bar(x, p_num_of_vol_04, linewidth=3.0, label="latent_04", color='none', edgecolor='blue')
	plt.bar(x, p_num_of_vol_08, linewidth=2.6, label="latent_08", color='none', edgecolor='red')
	plt.bar(x, p_num_of_vol_16, linewidth=2.2, label="latent_16", color='none', edgecolor='green')
	plt.bar(x, p_num_of_vol_32, linewidth=1.8, label="latent_32", color='none', edgecolor='orange')
	ax3.set_xticklabels(['0', '1', '2', '3'])
	plt.legend()

	
	ax1.legend(loc='upper right', fontsize=8)
	ax2.legend(loc='upper right', fontsize=8)
	ax3.legend(loc='upper right', fontsize=8)

	ax1.set(xlabel='Volume', ylabel='Proportion')
	ax2.set(xlabel='Volume', ylabel='Proportion')
	ax3.set(xlabel='Volume', ylabel='Proportion')

	ax1.grid()
	ax2.grid()
	ax3.grid()

	box = ax1.get_position()
	box.x0 = box.x0 - 0.03
	box.x1 = box.x1 - 0.03
	box.y0 = box.y0 + 0.03
	box.y1 = box.y1 + 0.03
	ax1.set_position(box)

	box = ax2.get_position()
	box.x0 = box.x0 + 0.03
	box.x1 = box.x1 + 0.03
	box.y0 = box.y0 + 0.03
	box.y1 = box.y1 + 0.03
	ax2.set_position(box)


	box = ax3.get_position()
	box.x0 = box.x0 + 0.2
	box.x1 = box.x1 + 0.2
	box.y0 = box.y0 - 0.04
	box.y1 = box.y1 - 0.04
	ax3.set_position(box)


	
	plt.savefig(os.path.join(CODE_FOLDER, 'analysis_results/11_volume_hist/volume_histogram.png'), dpi=200)
	plt.show()
	
	


				


    




        
if __name__=="__main__":
    main()