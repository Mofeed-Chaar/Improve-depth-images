import cv2
import os
import glob
import config
import pandas as pd 
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


print(device)
IMAGE_HEIGHT = 320  # 1280 originally
IMAGE_WIDTH = 480  # 1918 originally



image_path = '/home/chaar/Desktop/extended/img/draft/test'
path_model = os.path.join('highst_images_acc','model-level-1','model', 'highst_acc_model.pt')
model = torch.load(path_model)
model = model.to(device)

model.eval()
image_name = os.path.join(image_path, 'munster_000000_000000_leftImg8bit.png')
image = cv2.imread(image_name)
print(image.shape)
image = cv2.resize(image, (IMAGE_WIDTH,IMAGE_HEIGHT))
image = image/256
print(image.shape)
print(type(image))

img = torch.from_numpy(image).float().to(device)

img = img.unsqueeze(axis=0)
img = img.permute(0,3,1,2)
print(img.size())

preds = model(img)*126
preds = torch.round(preds)#.to(torch.int8)
preds = torch.where(preds<0,0,preds)
preds = torch.where(preds>126,126,preds)

print(preds.size())      
#torch_tensor.detach().cpu().numpy()

preds = preds.squeeze()
#preds = preds.permute(1,0)
pred_im = preds.detach().cpu().numpy()
pred_im =cv2.resize(pred_im,(2048,1024))
pred_im = np.left_shift(pred_im.astype(np.uint16), 8)
print(pred_im.shape)
cv2.imwrite(os.path.join(image_path,'pred.png'),pred_im)
#cv2.imshow('pred', pred_im) 
#cv2.waitKey(0)