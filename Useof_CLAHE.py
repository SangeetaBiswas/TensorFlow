#!/home/sangeeta/TensorFlow/venv/bin/python

#	========================================================
#	Purpose: To learn how to use CLAHE of OpenCV on an image
#	loaded by using TensorFlow + Keras and display it 
#	using matplotlib.	
#	--------------------------------------------------------
#	Note:
#		1.	Install OpenCV for python if it has not 
#			been installed yet.
#			$ pip3 install python-opencv 
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	27.7.2018
#	========================================================

#	1.	Import necessary modules.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

#	2. Declare constants
IMG_H = 64
IMG_W = 64
GRAYSCALE = False	#	Color Image --> False, 
					#	GrayScale Image --> True

#	3.	Load an image file.
imgFile = "/home/sangeeta/RetinaDatabase/retinaImage1.jpg"
img	=	tf.keras.preprocessing.image.load_img(
			imgFile,
			grayscale = GRAYSCALE,	
			target_size = None, #[IMG_H, IMG_W],	#	None / Tuple [h, w], e.g [64, 64]
			interpolation = 'bicubic'	#	Resampling methods such as 
										#	'bilinear', 'nearest', 'bicubic'
		)

#	4.	Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
#		object.
img = np.array(img)
claheImg = np.zeros(img.shape)
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
print('Max Pixel Value: {} & Min Pixel Value: {}'.format(img.max(), img.min()))
for i in range(3):
	claheImg[:,:,i] = clahe.apply(img[:,:,i])
print('Max Pixel Value: {} & Min Pixel Value: {}'.format(claheImg.max(), claheImg.min()))
claheImg = (claheImg - claheImg.min()) / (claheImg.max()-claheImg.min())
print('Max Pixel Value: {} & Min Pixel Value: {}'.format(claheImg.max(), claheImg.min()))

#	6.	Display loaded image after CLAHE.
fig = plt.figure(figsize = (10,10))

subPlt1 = fig.add_subplot(1, 2, 1)
subPlt1.set_title('Before applying CLAHE')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

subPlt2 = fig.add_subplot(1, 2, 2)
subPlt2.set_title('After applying CLAHE')
plt.xticks([])
plt.yticks([])
plt.imshow(claheImg)

plt.show()
