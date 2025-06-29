import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import random
import config
import pandas as pd 
from tqdm import tqdm
import os
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    test_dir,
    test_mskdir,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    test_ds = CarvanaDataset(
        image_dir=test_dir,
        mask_dir=test_mskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )


    return train_loader, val_loader,test_loader

def check_accuracy(loader, model, device="cuda"):
    pexil_prediction = 0
    model.eval()
    value_of_pexils = 0
    
    print('validation')

    with torch.no_grad():
        loop = tqdm(loader)

        for x, y in loop:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds = torch.where(preds < 0 , 0, preds)
            preds = torch.where(preds > 126 , 126, preds)

            pexil_prediction += (abs(preds - y)).sum()
            value_of_pexils += y.sum()
            
            
           
    print(f'\ndifference in pexil_prediction are {pexil_prediction} and acc is {100-pexil_prediction/value_of_pexils*100:.2f}')


    model.train()
    return (100-pexil_prediction/value_of_pexils*100)

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    print(f'saving test image in folder {folder}')
    model.eval()
    loop = tqdm(loader)
    for idx, (x, y) in enumerate(loop):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.where(preds < 0 , 0, preds)
            preds = torch.where(preds > 126 , 126, preds)
                       
            
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
        torchvision.utils.save_image(x, f"{folder}/img_{idx}.png")

    model.train()
    
    
def get_data_dirs(number_images = None,test = 10):
    data_info = config.config('data.yaml')
    side_train = data_info.cam_side['train']
    side_test = data_info.cam_side['test']
    print(side_test)
    train_dirs = []
    mask_train_dirs = []
    val_dirs = []
    mask_val_dirs = []

    data_train = pd.read_csv('data_train.csv')
    data_val = pd.read_csv('data_val.csv')
    if (side_train == 'left'):
        train_dirs = list(data_train['img_train_left'])
        mask_train_dirs = list(data_train['mask_train'])

    elif (side_train == 'right'):
        train_dirs = list(data_train['img_train_right'])
        mask_train_dirs = list(data_train['mask_train'])

    elif (side_train == 'both'):
        train_dirs = list(data_train['img_train_right']) + list(data_train['img_train_left'])
        mask_train_dirs = list(data_train['mask_train'])
        mask_train_dirs += mask_train_dirs


    if (side_test == 'left'):
        val_dirs = list(data_val['img_val_left'])
        mask_val_dirs = list(data_val['mask_val'])

    elif (side_test == 'right'):
        val_dirs = list(data_val['img_val_right'])
        mask_val_dirs = list(data_val['mask_val'])

    elif (side_test == 'both'):
        val_dirs = list(data_val['img_val_right']) + list(data_val['img_val_left'])
        mask_val_dirs = list(data_val['mask_val'])
        mask_val_dirs += mask_val_dirs
    temp = list(zip(val_dirs,mask_val_dirs))
    random.shuffle(temp)
    test_dirs,mask_test_dirs = zip(*temp)
        
    if number_images is None:    
        return train_dirs, mask_train_dirs, val_dirs, mask_val_dirs,test_dirs[:test],mask_test_dirs[:test]
    else:
        return train_dirs[:number_images[0]], mask_train_dirs[:number_images[0]],val_dirs[:number_images[1]], mask_val_dirs[:number_images[1]],test_dirs[:test],mask_test_dirs[:test]
        
        
        
def get_data_dirs_mask(number_images = None,test = 10,path='Data/cityscapes/improve_depth_level_6/Data/cityscapes/depth/disparity_sequence_trainvaltest/disparity_sequence'):
    """
    This function using to prepare the datasets that we want to make the depth images as input
    """
    print('training to improve the missing information in Depth images')
    train_dirs = []
    mask_train_dirs = []
    val_dirs = []
    mask_val_dirs = []

    data_train = pd.read_csv('data_train.csv')
    data_val = pd.read_csv('data_val.csv')
    
    
    train_dirs = list(data_train['mask_train'])
    mask_train_dirs = [update_mask_name(path,dirs) for dirs in train_dirs]





    
    val_dirs = list(data_val['mask_val'])
    mask_val_dirs = [update_mask_name(path,dirs) for dirs in val_dirs]


    temp = list(zip(val_dirs,mask_val_dirs))
    random.shuffle(temp)
    test_dirs,mask_test_dirs = zip(*temp)
        
    if number_images is None:    
        return train_dirs, mask_train_dirs, val_dirs, mask_val_dirs,test_dirs[:test],mask_test_dirs[:test]
    else:
        return train_dirs[:number_images[0]], mask_train_dirs[:number_images[0]],val_dirs[:number_images[1]], mask_val_dirs[:number_images[1]],test_dirs[:test],mask_test_dirs[:test]
        
        
        
def update_mask_name(path='Data/cityscapes/improve_depth_level_5/Data/cityscapes/depth/disparity_sequence_trainvaltest/disparity_sequence',file_dir='Data/cityscapes/depth/disparity_sequence_trainvaltest/disparity_sequence/val/frankfurt/frankfurt_000000_000283_disparity.png'):
    file_name = file_dir.split(os.path.sep)[-3:]
    all_file_name = os.path.sep.join(file_name)
    file_name = os.path.join(path,all_file_name)
    return file_name


if __name__=='__main__':
    update_mask_name()
