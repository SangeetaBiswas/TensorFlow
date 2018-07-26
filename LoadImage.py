#!/home/sangeeta/Tensorflow/bin/python

#	========================================================
#	Purpose: To learn how to load an image using
#	TensorFlow + Keras	and display it using matplotlib.		
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	2.7.2018
#	========================================================

#	1.	Import necessary modules.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#	2. Declare constants
IMG_H = 64
IMG_W = 64
GRAYSCALE = False	#	Color Image --> False, 
					#	GrayScale Image --> True

#	3.	Load an image file.
imgFile = "/home/sangeeta/retinaImage1.jpg"
img	=	tf.keras.preprocessing.image.load_img(
			imgFile,
			grayscale = GRAYSCALE,	
			target_size = None,	#	[IMG_H, IMG_W],	#	None / Tuple [h, w], e.g [64, 64]
			interpolation = 'bicubic'	#	Resampling methods such as 
										#	'bilinear', 'nearest', 'bicubic'
		)

#	4.	Separate Channels
imgAr = np.array(img)
imgR = imgAr[:,:,0]
imgG = imgAr[:,:,1]
imgB = imgAr[:,:,2]

#	5.	Get loaded image's information.
print('(Width, Height): {}'.format(img.size))
print('Shape: {}, Max Val: {}, Min Val: {}'.format(imgAr.shape, np.amax(imgAr), np.amin(imgAr)))
print('R: {}, G: {}, B: {}'.format(imgR.shape, imgG.shape, imgB.shape))

#	6.	Display loaded image.
fig = plt.figure(figsize=(20, 20))

subPlt1 = fig.add_subplot(2, 3, 1)
subPlt1.set_title('RGB')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

subPlt2 = fig.add_subplot(2, 3, 2)
subPlt2.set_title('Red Channel')
plt.xticks([])
plt.yticks([])
plt.imshow(imgR, cmap = plt.get_cmap('gray'))

subPlt3 = fig.add_subplot(2, 3, 3)
subPlt3.set_title('Green Channel')
plt.xticks([])
plt.yticks([])
plt.imshow(imgG, cmap = plt.get_cmap('gray'))

subPlt4 = fig.add_subplot(2, 3, 4)
subPlt4.set_title('Blue Channel')
plt.xticks([])
plt.yticks([])
plt.imshow(imgB, cmap = plt.get_cmap('gray'))
plt.show()
