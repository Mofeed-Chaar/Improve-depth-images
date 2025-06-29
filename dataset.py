import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

import glob
import cv2

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None,normalize = 126):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize = normalize

 
        

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        try:
            img_path = self.image_dir[index]
            mask_path = self.mask_dir[index]
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            mask = mask[:,:,0]#/126
            #mask[mask == 255.0] = 1.0

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            if self.normalize is not None:
        	    mask = mask/self.normalize
        except Exception as e:
            print(f"Error from file {mask_path}: {e}")

        return image, mask

if __name__ == '__main__':
	img_dir = '/media/chaar/INTENSO1/My-Project/My-work/Data/cityscapes/depth/disparity_sequence_trainvaltest/disparity_sequence/test/berlin'
	paths_img = glob.glob(os.path.join(img_dir,'*.png'))
	msk_dir = '/media/chaar/INTENSO1/My-Project/My-work/Data/cityscapes/depth/disparity_sequence_trainvaltest/disparity_sequence/test/berlin'
	
	test  = CarvanaDataset(paths_img,paths_img)
	image,mask = test[0]
	print(len(test))
	print(mask.shape)
	print(mask.max())
	#print(mask[40:60,40:60,0])
	

