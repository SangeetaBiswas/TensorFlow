#!/home/sangeeta/Tensorflow/bin/python

__version__ = 'v1'

#	========================================================
#	Pupose:	To store preprocessed image data and labels
#	by splitting into two sets so that they can be loaded
#	during parameter tuning of a NN based multiclass 
#	classifier.
#	--------------------------------------------------------
#	Database: EBD_RET
#			*** 110 folders
#			*** Each folder has 6-12 retina images.
#	
#	Training Set:	5 randomly selected images from folder.
#	Test Set:	Rest of the 1-7 images from each folder.
#
#	Class Labels: 0,1,2,3........,110
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	18.7.2018
#	========================================================

#	1. Import necessary modules.
import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import operator as opr

#	2.	Declare constants.
P_N = 110
IMG_H = 128
IMG_W = 128
GRAYSCALE = False	#	Color Image --> False, GrayScale Image --> True
if GRAYSCALE:
	CH_NO = 1
else:
	CH_NO = 3

M = 5; N = 'all' #	'all' / 1 

#	3.	Set path of database.
imgDir = "/media/sangeeta/HDPF-UT/Database/EBD_RET/"
dataDir = "/home/sangeeta/Programming/TensorFlow/"

#	4.	Define a function for splitting images of
#		each class into two sets randomly.
def getImgIndices(imgList, m, n):

	#	For imageList1
	allIndices = range(len(imgList))
	indices1 = random.sample(allIndices, m)
	imgList1 = opr.itemgetter(*indices1)(imgList)

	#	For imageList2
	A = set(allIndices)
	B = set(indices1)
	indices2 = list(A.difference(B))
	if (n == 'all'):
		imgList2 = opr.itemgetter(*indices2)(imgList)
	else:
		indices3 = random.sample(indices2, n)
		imgList2 = opr.itemgetter(*indices3)(imgList)

	if (len(indices2) == 1 or n == 1):
		imgList2 = [imgList2]	#	Otherwise, each character of the
								#	filename will be considered as a
								#	single element of a list.
	return imgList1, imgList2

#	5.	Make empty lists to hold all images and labels.
trnImgSet = []
trnLblSet = []
tstImgSet = []
tstLblSet = []

#	6.	Load all images and labels of a specific person
#		into a list of a specific set (training/test set).
def loadImgLabel(ptDirPath, imgList, imgSet, lblSet, label):
	for imgFileName in imgList:
		imgFile = ptDirPath + imgFileName
		img	=	tf.keras.preprocessing.image.load_img(
					imgFile,
					grayscale = GRAYSCALE,	
					target_size = [IMG_H, IMG_W],	#	'None' / Tuple [h, w], e.g [64, 64].
					interpolation = 'bicubic'	#	Resampling methods such as 
												#	'bilinear', 'nearest', 'bicubic'
				)
		imgSet.append(img.getdata())
		lblSet.append(label)

#	7.	Load all images and labels of all persons
#		into lists of training and test sets.
for dirPath, patientDirList, fileList in os.walk(imgDir):
	label = -1
	for patientDir in patientDirList[:P_N]:
		label += 1
		ptDir = dirPath + ''.join(patientDir) + '/'
		for ptDirPath, ptSubDir, imgList in os.walk(ptDir):
			ptTrnList, ptTstList = getImgIndices(imgList, M, N)
			loadImgLabel(ptDirPath, ptTrnList, trnImgSet, trnLblSet, label)
			loadImgLabel(ptDirPath, ptTstList, tstImgSet, tstLblSet, label)
			print('{}. {}: {} ({}, {})'.format(label, ptDirPath,
											len(imgList),len(trnImgSet),
											len(tstImgSet)))

#	8.	Convert lists into NumPy arrays.
lTrn = len(trnImgSet)
lTst = len(tstImgSet)
trnImgSet = np.reshape(np.array(trnImgSet), (lTrn, IMG_H,IMG_W, CH_NO))
trnLblSet = np.array(trnLblSet)
tstImgSet = np.reshape(np.array(tstImgSet), (lTst, IMG_H,IMG_W, CH_NO))
tstLblSet = np.array(tstLblSet)

print('Shape of trnImgSet: {}'.format(trnImgSet.shape))
print('Shape of trnLblSet: {}'.format(trnLblSet.shape))
print('Shape of tstImgSet: {}'.format(tstImgSet.shape))
print('Shape of tstLblSet: {}'.format(tstLblSet.shape))

#	9. Preprocess Images.

#	9.1.	To keep only Green Channel.
trnImgSet = trnImgSet[:,:,:,1] 
tstImgSet = tstImgSet[:,:,:,1] 
print('After keeping green channel: ')
print('\ttrnImgSet: {}'.format(trnImgSet.shape))
print('\ttstImgSet: {}'.format(tstImgSet.shape))

#	9.2.	Since there is only green channel now, we need to
#		reshape arrays to ensure they have 4 dimensions.
CH_NO = 1	
trnImgSet = np.reshape(trnImgSet, (lTrn, IMG_H,IMG_W, CH_NO))
tstImgSet = np.reshape(tstImgSet, (lTst, IMG_H,IMG_W, CH_NO))
print('After reshaping: ')
print('\ttrnImgSet: {}'.format(trnImgSet.shape))
print('\ttstImgSet: {}'.format(tstImgSet.shape))

#	9.3.	Convert the array into float32 as opposed to uint8.
trnImgSet = trnImgSet.astype(np.float32)
tstImgSet = tstImgSet.astype(np.float32)

#	9.4.	Convert the pixel values from integers 0-255 
#			into floats 0-1.
trnImgSet /= 255
tstImgSet /= 255

#	10. Prepare Labels
trnLblSet = tf.keras.utils.to_categorical(trnLblSet, P_N)
tstLblSet = tf.keras.utils.to_categorical(tstLblSet, P_N)

#	11.	Save images and labels of training and test sets
#		into a .npz file.
saveFileName =	dataDir + 'EBD_RET_' + str(P_N) + '_' + \
				str(IMG_H) + '-by-' + str(IMG_W) + '_G.npz' 
np.savez(saveFileName, 
		trnImgSet = trnImgSet,
		trnLblSet = trnLblSet,
		tstImgSet = tstImgSet,
		tstLblSet = tstLblSet
)

#	12.	Check whether data was stored correctly by loading  
#		and displaying some randomly picked images.

#	12.1.	Load data.
loadFileName =	dataDir + 'EBD_RET_' + str(P_N) + '_' + \
				str(IMG_H) + '-by-' + str(IMG_W) + '_G.npz' 
X = np.load(loadFileName)
imgSet = X['trnImgSet']
lblSet = X['trnLblSet']
print('Shape of Loaded Image Array: {}'.format(imgSet.shape))

#	12.2.	Display Image.
fig = plt.figure(figsize = (20, 20))
indices = random.sample(range(imgSet.shape[0]), 4)
j = 1
for i in indices:
	subPlt = fig.add_subplot(1, 4, j)
	label = np.where(lblSet[i] == 1)[0][0]
	subPlt.set_title('Person-'+ str(label))
	j += 1
	plt.xticks([])
	plt.yticks([])
	plt.imshow(imgSet[i,:,:,0], cmap = plt.get_cmap('gray'))
plt.show()
