#!/home/sangeeta/TensorFlow/venv/bin/python

#	========================================================
#	Purpose: To display blood vessels predicted by
#	CNN_BloodVesselMarker. 
#	--------------------------------------------------------
#	Images were taken from: 
#		DRIVE/test database
#		20 images 
#		https://www.isi.uu.nl/Research/Databases/DRIVE/
#	--------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	26.7.2018 
#	========================================================

#	1.	Import necessary modules
import numpy as np
import matplotlib.pyplot as plt

#	2.	Declare constants
EPOCH_NO = 100
DIR = '/home/sangeeta/Programming/TensorFlow/'
DB = 'DRIVE+HRF+STARE+CHASEDB1'

#	3.	Load Images.
fileName = DIR + 'CNN-BVM_' + str(EPOCH_NO) + 'Epoch_' + DB
loadFileName = fileName + '.npz'
X = np.load(loadFileName)
tstInImgSet = X['retinaImgSet']
tstOutImgSet = X['bloodVesselImgSet']
predictedImgSet = X['predictedBVImgSet']

#	4.	Display Images	
def displayImg(indices):
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

print('Displayed Images will be saved after figure will be closed.')
j = 0
for i in range(5):
	fig = plt.figure(figsize = (20, 20))
	displayImg(range(j, j+4))
	j += 4

	#	Save Displayed Figure.
	saveFigName = fileName + '_PredictedBV_' + str(j) + '-' + str(j+4) + '.jpg'
	fig.savefig(saveFigName)
