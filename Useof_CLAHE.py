#!/home/sangeeta/Tensorflow/bin/python

#	========================================================
#	Purpose: To learn how to use CLAHE of OpenCV on an image
#	loaded by using TensorFlow + Keras and display it 
#	using matplotlib.		
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
imgFile = "/home/sangeeta/retinaImage1.jpg"
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
claheImg[:,:,0] = clahe.apply(img[:,:,0])
claheImg[:,:,1] = clahe.apply(img[:,:,1])
claheImg[:,:,2] = clahe.apply(img[:,:,2])

claheImg = claheImg.astype(np.float32)
claheImg /= 255

#	6.	Display loaded image after CLAHE.
fig = plt.figure(figsize = (10,10))

subPlt1 = fig.add_subplot(1, 2, 1)
subPlt1.set_title('RGB')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

subPlt2 = fig.add_subplot(1, 2, 2)
subPlt2.set_title('CLAHE')
plt.xticks([])
plt.yticks([])
plt.imshow(claheImg)

plt.show()
