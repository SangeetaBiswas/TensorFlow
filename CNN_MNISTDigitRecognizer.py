#!/home/sangeeta/TensorFlow/venv/bin/python

#	========================================================
#	Purpose: To build a Convolutional Neural Network (CNN) 
#	in order to classify 10 English digits 
#	[i.e., 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] of MNIST Digit 
#	dataset and display some randomly selected didgits
#	from the test data set with true label and predicted 
#	label.
#	--------------------------------------------------------
#	Help was taken from 1-mnist-with-keras.ipynb 
#	available at:
# 	https://colab.research.google.com/github/tensorflow/workshops/blob/master/notebooks/1-mnist-with-keras.ipynb#scrollTo=AwxNOsCMNNGd
#	--------------------------------------------------------
#	Distribution of digits in the training and test sets
#	in MNIST dataset:
#						TrainingSet		TestSet
#						-----------		----------
#				Zero:		5923			980
#				One:		6742			1135
#				Two:		5958			1032
#				Three:		6131			1010
#				Four:		5842			982
#				Five:		5421			892
#				Six:		5918			958
#				Seven:		6265			1028
#				Eight:		5851			974
#				Nine:		5949			1009
#				-----		-----			-----
#				Total		60000			10000
#	
#	--------------------------------------------------------
#	Output:
#	Even by one epoch, it is possible to get
#		1.	training accuracy: 0.9513
#		2.	test accuracy: 0.9816	
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	17.7.2018, 31.10.2018
#	========================================================

#	1.	Import necessary modules.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

#	2.	Declare constants.
CLASS_NO = 10
CONV_LAYER = 1
FILTER_NO = [16, 6]
KERNEL_SZ = [3, 3]
STRIDE = [1, 1]
FC_LAYER = 3
HUNIT_NO = [512, 100, 50]
EPOCH_NO = 1
CH_NO = 1
N = 1000
K = 5

#	3.	Download the MNIST dataset.
(trnImgSet, trnLblSet), (tstImgSet, tstLblSet) = tf.keras.datasets.mnist.load_data()

#	4. Reformat the numpy arrays holding images.
#	4.1.	Know the number of images in the training
#			and test set.
lTrn, imgH, imgW = trnImgSet.shape
lTst = tstImgSet.shape[0]
print("Size of Training Set: {}" .format(lTrn))	
print("Size of Test Set: {}" .format(lTst))
print("Size of Each Image: {}-by-{}" .format(imgH, imgW))

#	4.2.	Reshape arrays of train images and test images,  
#			so that it can be passed to a Convolution-Pooling 
#			layer made by Conv2D().
trnImgSet = np.reshape(trnImgSet, (lTrn, imgH, imgW, CH_NO))
tstImgSet = np.reshape(tstImgSet, (lTst, imgH, imgW, CH_NO))

#	4.3.	Convert arrays to float32 as opposed to uint8.
trnImgSet = trnImgSet.astype(np.float32)
tstImgSet = tstImgSet.astype(np.float32)

#	4.4.	Convert the pixel values from integers 0-255 
#			to floats 0-1.
trnImgSet /= 255
tstImgSet /= 255

#	5.	Reformat the labels.
#	========================================================
#	Convert the labels from an integer format to an one-hot 
#	encoding , e.g.,
#		2 ---> "0, 0, 1, 0, 0, 0, 0, 0, 0, 0"
#		7 ---> "0, 0, 0, 0, 0, 0, 0, 1, 0, 0"
#	========================================================
trnLblSet_OHE  = tf.keras.utils.to_categorical(trnLblSet, CLASS_NO)
tstLblSet_OHE  = tf.keras.utils.to_categorical(tstLblSet, CLASS_NO)

#	6.	Build a model.
#	6.1.	Take an empty model to which layers can be added.
model = tf.keras.Sequential()

#	6.2. Add Convolution and Pooling layers.
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
			input_shape = (imgH, imgW, CH_NO)))
	else:
		model.add(tf.keras.layers.Conv2D(
			filters = FILTER_NO[i], 
			kernel_size = KERNEL_SZ[i], 
			strides = STRIDE[i],  
			padding = 'valid',
			use_bias = True,
			activation = tf.nn.relu))
	model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))

#	6.3.	Prepare output of convolution layers for  
#			Fully-Connected (FC)layers.
model.add(tf.keras.layers.Flatten())

#	6.4.	Add Fully-Connected (FC) layers.
for i in range(FC_LAYER):
	model.add(tf.keras.layers.Dense(HUNIT_NO[i], activation = tf.nn.relu))

#	6.5.	Add output layer.
model.add(tf.keras.layers.Dense(CLASS_NO, activation = tf.nn.softmax))

#	6.6.	Compile and print out a summary of the model.
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

model.summary()

#	7.	Train the designed model.
model.fit(trnImgSet[:N], trnLblSet_OHE[:N] , epochs = EPOCH_NO)

#	8.	Test the performance of the trained model.
loss, accuracy = model.evaluate(tstImgSet, tstLblSet_OHE )
print('Test accuracy: {}'.format(accuracy))

#	9.	Find the correct answer.
predict = model.predict(tstImgSet)
predicLabel = np.argmax(predict, axis=1)
print('True Label: {} Predicted Label: {}\n'.format(tstLblSet, predicLabel))

#	10.	Display 25 randomly selected digits from the test set.
for k in range(K):
	indices = random.sample(range(tstImgSet.shape[0]), 25)
	figTitle = 'Digits of Test Set'
	fig = plt.figure(figTitle, figsize = (20,20))

	j = 1
	for i in indices:
		fig.add_subplot(5, 5, j)
		j += 1
		title =	'[TL: ' + str(tstLblSet[i]) \
				+ '] [PL: ' + str(predicLabel[i]) + ']'
		plt.title(title)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(tstImgSet[i,:,:,0], cmap = plt.get_cmap('gray'))
 
	plt.show()

plt.close()

