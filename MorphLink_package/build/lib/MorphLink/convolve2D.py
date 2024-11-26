import numpy as np
import scipy.stats

def convolve2D(image, kernel, padding=1, strides=1, padding_value=0):
	# Cross Correlation
	kernel = np.flipud(np.fliplr(kernel))
	# Gather Shapes of Kernel + Image + Padding
	xKernShape = kernel.shape[0]
	yKernShape = kernel.shape[1]
	xImgShape = image.shape[0]
	yImgShape = image.shape[1]
	# Shape of Output Convolution
	xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
	yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
	output = np.zeros((xOutput, yOutput))
	# Apply Equal Padding to All Sides
	if padding != 0:
		imagePadded = np.ones((image.shape[0] + padding*2, image.shape[1] + padding*2))*padding_value
		imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
	else:
		imagePadded = image
	# Iterate through image
	for y in range(image.shape[1]):
		# Exit Convolution
		if y > image.shape[1] - yKernShape:
			break
		# Only Convolve if y has gone down by the specified Strides
		if y % strides == 0:
			for x in range(image.shape[0]):
				# Go to next row once kernel is out of bounds
				if x > image.shape[0] - xKernShape:
					break
				try:
					# Only Convolve if x has moved by the specified Strides
					if x % strides == 0:
						output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape])
				except:
					break
	return output

def convolve2D_refine(image, kernel=[[1,1,1], [1,1,1], [1,1,1]], threshold=4, padding=1, strides=1, padding_value=0):
	# Cross Correlation
	kernel = np.flipud(np.fliplr(kernel))
	# Gather Shapes of Kernel + Image + Padding
	xKernShape = kernel.shape[0]
	yKernShape = kernel.shape[1]
	xImgShape = image.shape[0]
	yImgShape = image.shape[1]
	# Shape of Output Convolution
	xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
	yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
	output = np.zeros((xOutput, yOutput))
	# Apply Equal Padding to All Sides
	if padding != 0:
		imagePadded = np.ones((image.shape[0] + padding*2, image.shape[1] + padding*2))*padding_value
		imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
	else:
		imagePadded = image
	# Iterate through image
	for y in range(image.shape[1]):
		# Exit Convolution
		if y > image.shape[1] - yKernShape:
			break
		# Only Convolve if y has gone down by the specified Strides
		if y % strides == 0:
			for x in range(image.shape[0]):
				# Go to next row once kernel is out of bounds
				if x > image.shape[0] - xKernShape:
					break
				try:
					# Only Convolve if x has moved by the specified Strides
					if x % strides == 0:
						tmp=scipy.stats.mode(imagePadded[x: x + xKernShape, y: y + yKernShape], axis=None)
						# output[x, y] = [imagePadded[x, y], tmp.mode[0]][tmp.count[0]>=threshold]
						output[x, y] = [imagePadded[x, y], tmp.mode][tmp.count>=threshold]
				except:
					break
	return output



def convolve2D_clean(image, kernel, threshold, padding=1, strides=1, padding_value=0):
	# Cross Correlation
	kernel = np.flipud(np.fliplr(kernel))
	# Gather Shapes of Kernel + Image + Padding
	xKernShape = kernel.shape[0]
	yKernShape = kernel.shape[1]
	xImgShape = image.shape[0]
	yImgShape = image.shape[1]
	# Shape of Output Convolution
	xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
	yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
	output = np.zeros((xOutput, yOutput))
	# Apply Equal Padding to All Sides
	if padding != 0:
		imagePadded = np.ones((image.shape[0] + padding*2, image.shape[1] + padding*2))*padding_value
		imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
	else:
		imagePadded = image
	# Iterate through image
	for y in range(image.shape[1]):
		# Exit Convolution
		if y > image.shape[1] - yKernShape:
			break
		# Only Convolve if y has gone down by the specified Strides
		if y % strides == 0:
			for x in range(image.shape[0]):
				# Go to next row once kernel is out of bounds
				if x > image.shape[0] - xKernShape:
					break
				try:
					# Only Convolve if x has moved by the specified Strides
					if x % strides == 0:
						output[x, y] = 1*((kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()>=threshold)
				except:
					break
	return output





def convolve2D_nbr_num(image, center, nbr, kernel=[[1,1,1], [1,1,1], [1,1,1]], padding=1, strides=1, padding_value=-1):
	# Cross Correlation
	kernel = np.flipud(np.fliplr(kernel))
	# Gather Shapes of Kernel + Image + Padding
	xKernShape = kernel.shape[0]
	yKernShape = kernel.shape[1]
	xImgShape = image.shape[0]
	yImgShape = image.shape[1]
	# Shape of Output Convolution
	xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
	yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
	output = np.zeros((xOutput, yOutput))
	# Apply Equal Padding to All Sides
	if padding != 0:
		imagePadded = np.ones((image.shape[0] + padding*2, image.shape[1] + padding*2))*padding_value
		imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
	else:
		imagePadded = image
	# Iterate through image
	for y in range(image.shape[1]):
		# Exit Convolution
		if y > image.shape[1] - yKernShape:
			break
		# Only Convolve if y has gone down by the specified Strides
		if y % strides == 0:
			for x in range(image.shape[0]):
				# Go to next row once kernel is out of bounds
				if x > image.shape[0] - xKernShape:
					break
				try:
					# Only Convolve if x has moved by the specified Strides
					if x % strides == 0:
						if imagePadded[x, y]==center:
							output[x, y] = np.sum(imagePadded[x: x + xKernShape, y: y + yKernShape]==nbr)
						else: 
							output[x, y]=0
				except:
					break
	return output
