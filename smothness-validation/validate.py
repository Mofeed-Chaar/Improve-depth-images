import glob
import os
import cv2
import numpy as np
import tqdm
print('done')

path_images = '/mnt/E/My-Work/Improve_cityscape/Depth/Data/cityscapes/improve_depth_Level_1/Data/cityscapes/depth/disparity_sequence_trainvaltest/disparity_sequence'
print('working in folder:')
print(path_images)
print(os.path.isdir(path_images))
pathes = glob.glob(os.path.join(path_images,'*','*','*.png'))

def laplacian_variance(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return lap.var()

# --- 2. Tenengrad Measure (Gradient Energy) ---
def tenengrad(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sum(gx**2 + gy**2)

# --- Load image in grayscale ---
def load_image_gray_cv2(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)



#-----------------------------------main---------------------------------------------------------------

def main():
	loop = tqdm.tqdm(pathes)
	sum_lap=0
	sum_ten = 0

	for path in loop:
		img = load_image_gray_cv2(path)
		ten = tenengrad(img)
		lap = laplacian_variance(img)
		
		sum_lap+=lap
		sum_ten+=ten
		#break

	print(f'laplacian var = {sum_lap/len(pathes)}')

	print(f'tenengrad = {sum_ten/len(pathes)}')

if __name__=='__main__':
	main()