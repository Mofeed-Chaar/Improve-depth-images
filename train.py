import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from model import UNET
import pandas as pd
import os
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_data_dirs,
    get_data_dirs_mask,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(device=None))
print(f'Number of workers:{os.cpu_count()}')
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = os.cpu_count()
IMAGE_HEIGHT = 320  # 1280 originally
IMAGE_WIDTH = 480  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR,TRAIN_MASK_DIR,VAL_IMG_DIR,VAL_MASK_DIR,TESR_DIR,TEST_MASK_DIR =  get_data_dirs_mask(number_images = None,test = 16)# if you want train for new model  importt get_data_dirs(number_images = None,test = 16)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    num = 0
    loss_Mean = 0

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        #print(data.size())
        
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast(DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss_Mean += loss
            num +=1

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    loss_Mean /=num
    return loss_Mean


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    print(summary(model, (3, IMAGE_WIDTH, IMAGE_HEIGHT)))
   
    loss_fn = nn.MSELoss()#nn.BCEWithLogitsLoss()#nn.MSELoss()#
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader,test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TESR_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler(DEVICE)
    
    results = {'accuracy':[],'loss':[]}
    pd_results = None
    num = 1
    highst_acc = 0.00
    epoch_highst_acc = 0
    highst_model = model

    for epoch in range(NUM_EPOCHS):
        print(f'epoch Number {num}/{NUM_EPOCHS}')
        loss_Mean=train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        #save_checkpoint(checkpoint)

        # check accuracy
        accuracy= check_accuracy(val_loader, model, device=DEVICE)
        if highst_acc < float("{:.2f}".format(accuracy.cpu())):
            highst_acc = float("{:.2f}".format(accuracy.cpu()))
            torch.save(model, 'model/highst_acc_model.pt')
            epoch_highst_acc = num
            highst_model = model
        results['accuracy'].append(float("{:.2f}".format(accuracy.cpu()))) #float("{:.2f}".format(dice.cpu()))
        results['loss'].append(float("{:.4f}".format(loss_Mean.cpu())))
        print(f'loss function {float("{:.4f}".format(loss_Mean.cpu()))}')

        # print some examples to a folder
        if num % 5 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            save_predictions_as_imgs(
                test_loader, model, folder="saved_images/", device=DEVICE
                )
            torch.save(model, 'model/model_epoch_{:.0f}.pt'.format(num))
        
        num +=1
        print('\n\n\n')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)
    torch.save(model, 'model/model_final.pt')
    print(f'highst accuracy {highst_acc} in epoch number {epoch_highst_acc}')
    pd_results = pd.DataFrame.from_dict(results)
    pd_results.to_csv('results.csv')
    save_predictions_as_imgs(
        test_loader, highst_model, folder="highst_images_acc/", device=DEVICE
        )



if __name__ == "__main__":
    main()