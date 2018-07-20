#!/home/sangeeta/Tensorflow/bin/python

__version__ = 'v1'

#	========================================================
#	Purpose: To make a multi classifier using a CNN 
#	to recognize a person by retinal images. 
#	--------------------------------------------------------
#	Database: EBD_RET
#			*** 110 folders
#			*** Each folder has 6-12 images.
#
#	Training Set:	5 randomly selected images from each 
#					patient.
#	Test Set:	1-7 randomly selected images from each 
#				patient.
#	--------------------------------------------------------	
#	Outputs:
#		a.	For 28-by-28 images:
#			train acc: 0.6182	test acc: 0.35	
#		b.	For 64-by-64 images:
#			train acc: 0.8800	test acc: 0.49	
#		c.	For 128-by-128 images:
#			train acc: 0.9364	test acc: 0.50				
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	18.7.2018
#	========================================================

#	1.	Import necessary modules.
import tensorflow as tf
import numpy as np

#	2.	Declare constants.
P_N = 110
CONV_LAYER = 1
FILTER_NO = [16]
KERNEL_SZ = [3]
STRIDE = [1]
FC_LAYER = 3
HUNIT_NO = [512, 100, 50]
EPOCH_NO = 200
IMG_H = 64
IMG_W = 64
CH_NO = 1

#	3.	Load images and labels of training and test sets.
dataDir = "/home/sangeeta/Programming/TensorFlow/"
loadFileName =	dataDir + 'EBD_RET_' + str(P_N) + '_' + \
				str(IMG_H) + '-by-' + str(IMG_W) + '_G.npz' 
X = np.load(loadFileName)
trnImgSet = X['trnImgSet']
trnLblSet = X['trnLblSet']
tstImgSet = X['tstImgSet']
tstLblSet = X['tstLblSet']

#	4.	Build the model
#	===============================================================================
#	A simple CNN is designed using the Keras Sequential API which has:
#		a)	specific number of convolution layers.
#		b)	specific number of fully connected (FC) layers.
#		c)	different number of units in different hidden layers.
#		d)	all layers except the output layer use the ReLU activation function.
#		e)	the output layer uses softmax function.
#		f)	the categorical crossentropy loss function, and the RMSProp
#			optimizer are used.
#	===============================================================================

#	4.1.	Take an empty model to which layers can be added.
model = tf.keras.Sequential()

#	4.2.	Add Convolution and Pooling Layers
for i in range(CONV_LAYER):
	if (i == 0):
		#	The first layer in a Sequential model must have 
		#	an 'input_shape' argument.
		model.add(tf.keras.layers.Conv2D(
			filters = FILTER_NO[i], 
			kernel_size = KERNEL_SZ[i], 
			strides = STRIDE[i],  
			padding = 'valid',
			use_bias = True,
			activation = tf.nn.relu,
			input_shape = (IMG_H, IMG_W, CH_NO)))
	else:
		model.add(tf.keras.layers.Conv2D(
			filters = FILTER_NO[i], 
			kernel_size = KERNEL_SZ[i], 
			strides = STRIDE[i],  
			padding = 'valid',
			use_bias = True,
			activation = tf.nn.relu))
	model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))

#	4.3.	Prepare output of convolution layers for FC Layers
model.add(tf.keras.layers.Flatten())

#	4.4.	Add FC Layers.
for i in range(FC_LAYER):
	model.add(tf.keras.layers.Dense(HUNIT_NO[i], activation = tf.nn.relu))
	model.add(tf.keras.layers.Dropout(0.33))

#	4.5.	Add Output Layer.
model.add(tf.keras.layers.Dense(P_N, activation = tf.nn.softmax))

#	4.6.	Compile and print out a summary of the model.
model.compile(	loss = 'categorical_crossentropy',
				optimizer = 'rmsprop',
				metrics = ['accuracy'])
model.summary()

#	6.	Train the CNN.
model.fit(trnImgSet, trnLblSet, epochs = EPOCH_NO)

#	8.	Test the performance of the CNN.
loss, accuracy = model.evaluate(tstImgSet, tstLblSet)
print('Test accuracy: {}'.format(accuracy))

