import cv2
import os
import pandas as pd 
import torch
from tqdm import tqdm
import sys
import json
from utils import (save_img,
                   get_predection,
                   correct_zeros_in_image)

path_saving_data = 'Data/improve_depth'

device = "cuda" if torch.cuda.is_available() else "cpu"

orginal_image_size=((1024,2048))
print(device)
IMAGE_HEIGHT = 320  # 1280 originally
IMAGE_WIDTH = 480  # 1918 originally
os.chdir('..')

csv_dirs = 'Data/cityscapes/depth'
# load data dataframe
csv_train_image = os.path.join(csv_dirs,'data_train.csv')
csv_test_image = os.path.join(csv_dirs,'data_test.csv')
csv_val_image = os.path.join(csv_dirs,'data_val.csv')

train_dirs = pd.read_csv(csv_train_image)
test_dirs = pd.read_csv(csv_test_image)
val_dirs = pd.read_csv(csv_val_image)

test_dirs.rename({test_dirs.columns[0]: train_dirs.columns[0], 
                  test_dirs.columns[1]: train_dirs.columns[1],
                  test_dirs.columns[2]: train_dirs.columns[2]}, 
                 axis=1, inplace=True)

val_dirs.rename({val_dirs.columns[0]: train_dirs.columns[0], 
                  val_dirs.columns[1]: train_dirs.columns[1],
                  val_dirs.columns[2]: train_dirs.columns[2]}, 
                 axis=1, inplace=True)

frames = [train_dirs,test_dirs,val_dirs]

data = pd.concat(frames,ignore_index=True)



sys.path.append('model')





path_model = os.path.join('model', 'highst_acc_model.pt')
print(os.getcwd())
print(os.path.isfile(path_model))
print(path_model)
model = torch.load(path_model)
model = model.to(device)
model.eval()

#statistics
corrected_all_pexils = {'1-10':0,'11-20':0,'21-30':0,'31-40':0,'41-50':0,
                    '51-60':0,'61-710':0,'71-80':0,'81-90':0,'91-100':0,'101-110':0,'111-126':0}

same_all_zeros = 0 # average number of pexils that is zero and did not changed by improvment

len_all_zeros = 0 # average number of pexils that its value is zero in depth image befor improvement

number_of_images = len(data)

# Improve the depthe images in Cityscape dataset****************************************************
loop = tqdm(data['img_train_left'])
pred_im = None
#i=0
for i, dirs in enumerate(loop):
    image_dir = dirs
    pred_im = get_predection(image_dir,model=model,
                             image_size=(IMAGE_WIDTH,IMAGE_HEIGHT),
                             device=device)
    target_im_dir = data['mask_train'][i]
    target_im = cv2.imread(target_im_dir)
    
    improved_im, len_zeros, same_zeros,corrected_pexils = correct_zeros_in_image(target_im,pred_im)
    for key in corrected_all_pexils.keys():
        corrected_all_pexils[key] +=corrected_pexils[key]
    
    same_all_zeros += same_zeros
    
    len_all_zeros += len_zeros
    

    save_img(os.path.join(path_saving_data,target_im_dir),improved_im)
    
    #i+=1
    #if i ==5:
        #break



# process the statistics numbers****************************************************************
for key in corrected_all_pexils.keys():
    corrected_all_pexils[key] /=number_of_images
    
same_all_zeros /= number_of_images

len_all_zeros /=number_of_images

num_image_pexils = 1024*2048
json_result_file = {'corrected_all_pexils':corrected_all_pexils,'same_all_zeros':same_all_zeros,
                    'len_all_zeros':len_all_zeros,'num_image_pexils':num_image_pexils}

with open(os.path.join(path_saving_data,"json_result_file.json"), "w") as outfile: 
    json.dump(json_result_file, outfile)












