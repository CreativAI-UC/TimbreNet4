import os
from random import shuffle

path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),'datasets/audio_augmented_x10')


filenames = os.listdir(path)
shuffle(filenames)
train = filenames[0:38880]
val = filenames[38880:]

print(len(train))
print(len(val))

n = 0
for file in train:
	os.rename(os.path.join(path,file), os.path.join(os.path.join(os.path.dirname(path),'audioPianoTriadDataset/audio_train'),file))
	n = n + 1
	if n%100 == 0:
		print(n)

n = 0
for file in val:
	os.rename(os.path.join(path,file), os.path.join(os.path.join(os.path.dirname(path),'audioPianoTriadDataset/audio_val'),file))
	n = n + 1
	if n%100 == 0:
		print(n)

