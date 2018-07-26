#!/home/sangeeta/TensorFlow/venv/bin/python

#	================================================================================
#	Purpose: To segment blood vessels of retina images. 	
#	--------------------------------------------------------
#	Images were taken from 
#	1.	DRIVE database
#		40 images [20 training + 20 testing]
#		https://www.isi.uu.nl/Research/Databases/DRIVE/
#
#	2.	HRF Database
#		45 images [15 Healthy + 15 Glaucoma + 15 DR]
#		https://www5.cs.fau.de/research/data/fundus-images/
#	
#	3.	STARE Database
#		20 images
#		http://cecas.clemson.edu/~ahoover/stare/probing/index.html
#
#	4.	CHASE_DB1
#		28 images
#		https://www.kingston.ac.uk/faculties/science-engineering-and-computing/
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	25.7.2018 
#	================================================================================

#	1.	Import necessary modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time 
import h5py

#	2.	Declare constants
IMG_H = 256
IMG_W = 256
CONV_LAYER = 3		
FILTER_NO = [32, 64, 1]	
KERNEL_SZ = [11, 7, 5]		
STRIDE = [1, 1, 1]			
DROPOUT = 0.33
EPOCH_NO = 100
CH_NO = 3
DIR = '/home/sangeeta/Programming/TensorFlow/'
DB = 'DRIVE+HRF+STARE+CHASEDB1'

#	3.	Load Images.
loadFileName =	DIR + 'TrainTestSet_CNN-BVM_' + DB + '.npz'
X = np.load(loadFileName)
trnInImgSet = X['trnInImgSet']
trnOutImgSet = X['trnOutImgSet']
tstInImgSet = X['tstInImgSet']
tstOutImgSet = X['tstOutImgSet']

print('trnInImgSet.shape: {}'.format(trnInImgSet.shape))
print('trnOutImgSet.shape: {}'.format(trnInImgSet.shape))
print('tstInImgSet.shape: {}'.format(tstInImgSet.shape))
print('tstOutImgSet.shape: {}'.format(tstOutImgSet.shape))

#	4.	Build the model
#	4.1.	Take an empty model to which layers can be added.
model = tf.keras.Sequential()

#	4.2.	Add Convolution and Pooling Layers
for i in range(CONV_LAYER):
	if i == (CONV_LAYER - 1):
		activation = tf.nn.sigmoid
	else:
		activation = tf.nn.relu		
	if (i == 0):
		#	The first layer in a Sequential model must have 
		#	an 'input_shape' argument.
		model.add(tf.keras.layers.Conv2D(
			filters = FILTER_NO[i], 
			kernel_size = KERNEL_SZ[i], 
			strides = STRIDE[i],  
			padding = 'same',
			use_bias = True,
			activation = activation,
			input_shape = (IMG_H, IMG_W, CH_NO)))
	else:
		model.add(tf.keras.layers.Conv2D(
			filters = FILTER_NO[i], 
			kernel_size = KERNEL_SZ[i], 
			strides = STRIDE[i],  
			padding = 'same',
			use_bias = True,
			activation = activation))
	model.add(tf.keras.layers.Dropout(DROPOUT))

#	4.3.	Compile and print out a summary of the model.
model.compile(	loss = 'binary_crossentropy', 
		optimizer = 'rmsprop',
		metrics = ['accuracy'])
model.summary()

#	5.	Train the CNN.
startTime = time.time()
model.fit(trnInImgSet, trnOutImgSet, epochs = EPOCH_NO, batch_size = 2)
endTime = time.time()
print('Training time: {}'.format(endTime-startTime))

#	6.	Test the performance of the CNN.
loss, accuracy = model.evaluate(tstInImgSet, tstOutImgSet)
print('Test accuracy: {}'.format(accuracy))
predictedImgSet = model.predict(tstInImgSet)

#	7.	Display Some Randomly Selected Images	
print('Displaying Images....')
fig = plt.figure(figsize = (20, 20))
indices = random.sample(range(tstInImgSet.shape[0]), 4)
j = 1
for i in indices:
	subPlt1 = fig.add_subplot(4, 3, j)
	subPlt1.set_title('Original Image')
	j += 1
	plt.xticks([])
	plt.yticks([])
	plt.imshow(tstInImgSet[i])

	subPlt2 = fig.add_subplot(4, 3, j)
	subPlt2.set_title('Manually Segemented Image')
	j += 1
	plt.xticks([])
	plt.yticks([])
	plt.imshow(tstOutImgSet[i,:,:,0], cmap = plt.get_cmap('gray'))

	subPlt3 = fig.add_subplot(4, 3, j)
	subPlt3.set_title('NN Predicted Image')
	j += 1
	plt.xticks([])
	plt.yticks([])
	plt.imshow(predictedImgSet[i,:,:,0], cmap = plt.get_cmap('gray'))
plt.show()

#	8.	Save models and data arrays.
#	8.1.	Save the model
fileName = DIR + 'CNN-BVM_' + str(EPOCH_NO) + 'Epoch_' + DB
saveModelName = fileName + '.h5'
tf.keras.models.save_model(model, saveModelName)

#	8.2.	Save predicted images into a .npz file.
saveDataFileName =	fileName + '.npz' 
np.savez(saveDataFileName, 
	retinaImgSet = tstInImgSet,
	bloodVesselImgSet = tstOutImgSet,
	predictedBVImgSet = predictedImgSet
)



