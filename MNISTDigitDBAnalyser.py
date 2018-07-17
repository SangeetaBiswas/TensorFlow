#!/home/sangeeta/Tensorflow/bin/python

#	========================================================
#	Purpose: To load and analyze images of MNIST Digit 
#	database which has images of 10 English digits 
#	[i.e., 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] using TensorFlow
#	and Keras.
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	27.6.2018
#	========================================================

#	1.	Import necessary modules.
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

#	2.	Download the MNIST dataset.
# 	========================================================
#	Notes:
#	1.	MNIST database will be downloaded only once.
#	2.	If no path is given then 11MB MNIST data will be  
#		loaded in the home directory, 
#		~/.keras/datasets/mnist.npz which can be checked by:
#		$ ls -alh ~/.keras/datasets/mnist.npz
#		-rw-rw-r-- 1 sangeeta sangeeta 11M Jun 27 20:45 .keras/datasets/mnist.npz
# 	========================================================
print('Downloading MNIST Dataset if it is not already downloaded.')
dbPath = '/home/sangeeta/Programming/TensorFlow/mnist.npz'
(trnImgSet, trnLblSet), (tstImgSet, tstLblSet) = tf.keras.datasets.mnist.load_data(
													path = dbPath
												)

#	3.	Know the number of images in the training and test set.
lTrn, imgH, imgW = trnImgSet.shape
lTst = tstImgSet.shape[0]
print("Number of Images in the Training Set: {}" .format(lTrn))	
print("Number of Images in the Test Set: {}" .format(lTst))
print("Size of Each Image: {}-by-{}" .format(imgH, imgW))

#	4.	Get distribution of digit in the training
#		and test sets.	
digitLabel = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five',
				'Six', 'Seven', 'Eight', 'Nine', 'Ten']
trnDgt, trnDgtCount = np.unique(trnLblSet, return_counts = True)
print('Number of Images of Each Digit in the Training Set:')
for i in range(10):
	print('  {}: {}'.format(digitLabel[trnDgt[i]], trnDgtCount[i]))
print('Total Number of Images in the Training Set: {}'.format(np.sum(trnDgtCount)))

tstDgt, tstDgtCount = np.unique(tstLblSet, return_counts = True)
print('Number of Images of Each Digit in the Test Set:')
for i in range(10):
	print('  {}: {}'.format(digitLabel[tstDgt[i]], tstDgtCount[i]))
print('Total Number of Images in the Test Set: {}'.format(np.sum(tstDgtCount)))

#	5.	Display 30 images selected randomly from the 
#		training set.
#	========================================================
#	plt.imshow() is for displaying image on a figure, i.e, 
#	on a window. plt.show() is for displaying a figure. 
#	These two methods need to be used together, otherwise, 
#	no image is displayed in my machine i.e., Ubuntu 16.04.
#	========================================================
print('Displaying 30 images selected randomly from the training set.')
indices = random.sample(range(lTrn), 30)
i = 1
plt.figure(figsize = (10,10))
plt.suptitle('30 Images from MNIST Training Set', fontsize = 16, fontweight = 'bold')
for j in indices:
	plt.subplot(6,5,i)
	plt.xticks([])
	plt.yticks([])
	plt.xlabel(digitLabel[trnLblSet[j]])
	plt.imshow(trnImgSet[j], cmap = plt.cm.binary)
	i += 1
plt.show()
