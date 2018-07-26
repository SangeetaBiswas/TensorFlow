#!/home/sangeeta/TensorFlow/venv/bin/python

#	================================================================================
#	Purpose: To preprocess images for detecting blood 
#	vessels in a retina image. 	
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
import os
import matplotlib.pyplot as plt
import random

#	2.	Declare constants
IMG_H = 256
IMG_W = 256
DB = 'DRIVE+HRF+STARE+CHASEDB1'

#	3.	Load all images of Training set and Test Set into Numpy arrays.
#	3.1.	Define a function to load a single image.
def loadImage(imgFile, imgSet, graySc):
	img =	tf.keras.preprocessing.image.load_img(
			imgFile,
			grayscale = graySc,	
			target_size = [IMG_H, IMG_W],	
			interpolation = 'bicubic'	
		)
	imgSet.append(img.getdata())


#	3.2.	Define a function to load all images into a NumPy array
#			for a specific data set.
def makeImgSet(tgtDir, index, imgSet1, imgSet2):
	print('Loading data from {} has been started.'.format(tgtDir))
	imgDir = tgtDir + retDir[index]
	print(imgDir)
	for dirPath, subDir, imgList in os.walk(imgDir):
		for imgFileName in imgList:
			imgFile =  imgDir + imgFileName
			print(imgFile)
			loadImage(imgFile, imgSet1, False)
			imgFile = tgtDir + bldVesDir[index] + imgFileName[:header[index]] + ext[index]
			print(imgFile)
			loadImage(imgFile, imgSet2, True)

#	3.3.	Figure out varieties.
retDir = [	'images/', 'healthy/', 'glaucoma/', 'diabetic_retinopathy/', 
			'stare-images/', 'images/', 'images/']
bldVesDir = [	'1st_manual/', 'healthy_manualsegm/', 'glaucoma_manualsegm/', 
				'diabetic_retinopathy_manualsegm/', 'labels-vk/', 
				'segmented/', '1st_manual/']
header = [2, 2, 2, 2, 6, 9, 2]
ext = [	'_manual1.gif', '_h.tif', '_g.tif', '_dr.tif', 
		'.vk.ppm', '_1stHO.png', '_manual1.gif'] 

imgDir =	[	"/home/sangeeta/RetinaDatabase/DRIVE/training/",
				"/home/sangeeta/RetinaDatabase/HRF/",
				"/home/sangeeta/RetinaDatabase/HRF/",
				"/home/sangeeta/RetinaDatabase/HRF/",
				"/home/sangeeta/RetinaDatabase/STARE/",
				"/home/sangeeta/RetinaDatabase/CHASE_DB1/",
				"/home/sangeeta/RetinaDatabase/DRIVE/test/"
			]

#	3.4.	Load images by calling functions.
trnInImgSet = []
trnOutImgSet = [] 
tstInImgSet = []
tstOutImgSet = [] 

l = len(imgDir) - 1
for i in range(l):
	makeImgSet(imgDir[i], i, trnInImgSet, trnOutImgSet)
	print('Number of Loaded Images into Training Set: {}'.format(len(trnInImgSet)))

makeImgSet(imgDir[l], l, tstInImgSet, tstOutImgSet)

#	3.5.	Turn list into NumPy arrays and reshape arrays.
lTrn = len(trnInImgSet)
trnInImgSet = np.reshape(np.array(trnInImgSet), (lTrn, IMG_H, IMG_W, 3))
trnOutImgSet = np.reshape(np.array(trnOutImgSet), (lTrn, IMG_H, IMG_W, 1))

lTst = len(tstInImgSet)
tstInImgSet = np.reshape(np.array(tstInImgSet), (lTst, IMG_H, IMG_W, 3))
tstOutImgSet = np.reshape(np.array(tstOutImgSet), (lTst, IMG_H, IMG_W, 1))

print('trnInImgSet.shape: {}\ntrnOutImgSet.shape: {}'.format(trnInImgSet.shape, trnOutImgSet.shape))
print('tstInImgSet.shape: {}\ntstOutImgSet.shape: {}'.format(tstInImgSet.shape, tstOutImgSet.shape))
	
#	4.	Preprocess images.
#	4.1.	Convert the array to float32 as opposed to uint8
trnInImgSet = trnInImgSet.astype(np.float32)
trnOutImgSet = trnOutImgSet.astype(np.float32)
tstInImgSet = tstInImgSet.astype(np.float32)
tstOutImgSet = tstOutImgSet.astype(np.float32)

#	4.2.	Convert the pixel values from integers 0-255 
#		to floats 0-1
trnInImgSet /= 255
trnOutImgSet /=  255
tstInImgSet /= 255
tstOutImgSet /=  255

#	5.	Display some randomly selected images only to confirm everything is Okay.
def displayImg(imgSet1, imgSet2, title):
	indices = random.sample(range(imgSet1.shape[0]), 4)
	j = 1

	fig = plt.figure(figsize = (20, 20))
	plt.suptitle(title, fontsize = 16)
	for i in indices:
		subPlt = fig.add_subplot(2, 4, j)
		j += 1
		plt.xticks([])
		plt.yticks([])
		plt.imshow(imgSet1[i])
	for i in indices:
		subPlt = fig.add_subplot(2, 4, j)
		j += 1
		plt.xticks([])
		plt.yticks([])
		plt.imshow(imgSet2[i,:,:,0], cmap = plt.get_cmap('gray'))
	plt.show()

print('Displaying randomly selected images....')
displayImg(trnInImgSet, trnOutImgSet, 'Training Set')
displayImg(tstInImgSet, tstOutImgSet, 'Test Set')

#	6.	Save loaded images into a .npz file.
print('Saving images....')
saveFileName =	'/home/sangeeta/Programming/TensorFlow/TrainTestSet_CNN-BVM_' + DB + '.npz' 
np.savez(saveFileName, 
		trnInImgSet = trnInImgSet,
		trnOutImgSet = trnOutImgSet,
		tstInImgSet = tstInImgSet,
		tstOutImgSet = tstOutImgSet
)

