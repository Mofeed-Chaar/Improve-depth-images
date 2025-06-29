#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:18:35 2024

@author: chaar
"""
import os
import cv2
import torch
import numpy as np
import copy
#os.chdir('..')

def get_img_name(img_dir=None):
    image_name = '_'.join(img_dir.split('/')[-1].split('_'))
    return image_name

def get_path_name(path = None):
    path_name = os.path.join(*path.split('/')[:-1])
    return path_name

def mkdir(path = None):
    if not os.path.isdir(path):
        os.makedirs(path)
        
        
        
        
def save_img(path = None,img = None):
    image_name = get_img_name(path)
    dir_name = get_path_name(path) # directory of image
    mkdir(dir_name)
    img_dir = os.path.join(dir_name, image_name)
    
    cv2.imwrite(img_dir, img)
    
    
def get_predection(path=None,model=None,image_size=(),device='cuda'):
    
    image = cv2.imread(path)
    orginal_image=image.shape
    image = cv2.resize(image, image_size)
    image = image/256

    img = torch.from_numpy(image).float().to(device)

    img = img.unsqueeze(axis=0)
    img = img.permute(0,3,1,2)

    preds = model(img)*126
    preds = torch.round(preds).to(torch.int16)
    preds = torch.where(preds<0,0,preds)
    preds = torch.where(preds>126,126,preds)

    preds = preds.squeeze()
    pred_im = preds.detach().cpu().numpy()
    pred_im = cv2.resize(pred_im,(orginal_image[1],orginal_image[0])).astype(int)
    
    #pred_im = pred_im.astype(int)
    return pred_im

def correct_zeros_in_image(target_im=None,pred_im=None):
    same_zeros = 0
    corrected_pexils = {'1-10':0,'11-20':0,'21-30':0,'31-40':0,'41-50':0,
                        '51-60':0,'61-710':0,'71-80':0,'81-90':0,'91-100':0,'101-110':0,'111-126':0}
    zeros = np.where(target_im == 0)
    len_zeros = len(zeros[0])
    improved_im = copy.copy(target_im)
    
    
    for i in range(len_zeros):
        if pred_im[zeros[0][i],zeros[1][i]]==0:
            same_zeros +=1
            continue
        # counting the corrected pexils in the target image
        corrected_pexils[list(corrected_pexils.keys())[int((pred_im[zeros[0][i],zeros[1][i]]//10)-1)]] +=1
        improved_im[zeros[0][i],zeros[1][i]] = pred_im[zeros[0][i],zeros[1][i]]
    return improved_im, len_zeros, same_zeros,corrected_pexils
    


if __name__=='__main__':
    path = 'train/aachen/aachen_000062_000011_disparity.png'
    path_test = 'test/test2'
    print(get_path_name(path))
    #mkdir(path_test)
    help(cv2.imshow)
    
    