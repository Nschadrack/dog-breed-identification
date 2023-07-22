import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import wandb
import os
from dotenv import load_dotenv
import uuid
import gc
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler
 
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms
from config import config

from dataset_classes import TrainDogBreedDataset, TestDogBreedDataset
from utils import training, validating, test_model, calculate_accuracy, calculate_loss, load_model_checkpoint, save_model_checkpoint
from model import DogBreedNet, create_model


warnings.simplefilter('ignore')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("\nDevice: {}\n".format(device))


ROOT_DATA_DIR = 'data'
TEST_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'test')
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'train')

# reading the csv labels
labels_df = pd.read_csv(os.path.join(ROOT_DATA_DIR, 'labels.csv'))

# extracting unique class names(breed)
labels = labels_df["breed"].unique().tolist()

# mapping breed with numerical value raning from 0 to 119(since we have 120 breed/classes)
labels_map = { value: index for index, value in enumerate(labels)}
labels_value_to_breed = { index: value for index, value in enumerate(labels)}

# adding a new column to labels dataframe by mapping breed column with 
# its corresponding numerical label
labels_df['label'] = labels_df['breed'].map(labels_map)

# extracting the numerical labels from labels dataframe
labels_num = labels_df['label'].tolist()

# encoding cardinal labels to 1 hot encoding. This is the final encoding to be used in the model
labels = np.eye(120)[labels_num]


gc.collect()
train_transforms_args = [
    transforms.RandomResizedCrop(256),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
valid_transforms_args = [
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
train_dataset = TrainDogBreedDataset(root_dir=TRAIN_DATA_DIR, 
                                     partition="train",
                                    labels=labels,
                                    transform=train_transforms_args)

valid_dataset = TrainDogBreedDataset(root_dir=TRAIN_DATA_DIR, 
                                     partition="train",
                                    labels=labels,
                                    transform=valid_transforms_args,
                                    training_data=False)


# test_dataset = TestDogBreedDataset(root_dir=TEST_DATA_DIR, 
#                                      partition="test",
#                                      transform=[transforms.Resize((256, 256)),transforms.ToTensor()])


#  testing data loaders
valid_loader = DataLoader(dataset=valid_dataset,
                         batch_size=64,
                         shuffle=False,
                         num_workers=2)


gc.collect()
for X, y in valid_loader:
    X, y = X.to(device), y.to(device)

model = DogBreedNet(input_channels=3, num_classes=120, dense_dropout_p=.55).to(device)
# model = create_model(120, .55).to(device)
summary(model, input_size=X.shape[1:], batch_size=64)


# Defining Data loaders to be used for training and testing
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=config["batch_size"],
                         shuffle=True,
                         num_workers=4,
                         pin_memory=True)

valid_loader = DataLoader(dataset=valid_dataset,
                         batch_size=config["batch_size"],
                         shuffle=True,
                         num_workers=2,
                         pin_memory=True)

# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=conifg["batch_size"],
#                          shuffle=False,
#                          num_workers=2)


# model experimentation
# 1. defining optimizer
# 2. defining shedulers
# 3. defining scaler
# 4. configuring wandb
# 5. writing training loops
# 6. defining model checkpoint path

gc.collect()
torch.cuda.empty_cache()
# Defining optimizer
optimizer = optim.SGD(params=model.parameters(), 
                      lr=config['lr'], 
                      weight_decay=config['weight_decay'], 
                      momentum=0.9)

# Defining scheduler
# scheduler = StepLR(optimizer, base_lr=config['base_lr'], max_lr=config['max_lr'], step_size_up=20, 
#                      mode='triangular', cycle_momentum=False)
scheduler = StepLR(optimizer, step_size=50)

# Defining scaler
scaler = GradScaler()

# Configuring WANDB
load_dotenv(os.path.join(os.getcwd(), '.env'))
wandb.login(key=os.getenv("WANDB_KEY"))
run = wandb.init(
    name='{}'.format(str(uuid.uuid4())[:12]),
    reinit=True,
    project='Dog-breed-ID-Log-V5',
    config=config
)

# defining path where the model check point will be saved
# model_checkpoint_path = os.path.join(ROOT_DATA_DIR, "dog_breed_id_checkpoint.pth") # running on kaggle
model_checkpoint_path = os.path.join('checkpoints', "dog_breed_id_checkpoint_v4.pth") # running on AWS

# implementing training loop for experimentations
print("\n\n")
best_validation_accuracy = 0.0
for epoch in range(1, config['epochs'] + 1, 1):
    current_lr = optimizer.param_groups[0]['lr']
    train_accuracy, train_loss = training(model=model, optimizer=optimizer, train_data_loader=train_loader, scaler=scaler)
    
    with open("experiments_log_v4.text", "a") as f:
        train_line = "Epoch {}/{}:\nTrain Acc: {}%\t Training Loss: {}\t Learning Rate: {}\n".format(epoch, config['epochs'], round(train_accuracy, 4), round(train_loss,4), round(current_lr, 4))
        print(train_line)
        
        validation_accuracy, validation_loss = validating(model=model, valid_data_loader=valid_loader)

        valid_line = "Validation Acc: {}%\t Validation Loss: {}\t Learning Rate: {}\n".format(round(validation_accuracy, 4), round(validation_loss, 4), round(current_lr, 4))
        
        print(valid_line)

        f.write(train_line)
        f.write(valid_line)
        f.write("\n=======================================================================================================\n")
                                     
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": validation_loss,
            "Validation Accuracy": validation_accuracy,
            "Learning Rate": current_lr
        })
        
        if validation_accuracy >  best_validation_accuracy:
            save_model_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_acc=validation_accuracy, path=model_checkpoint_path, scheduler=scheduler)
            best_validation_accuracy = validation_accuracy
            
    #         saving checkpoint in wandb
            wandb.save('dog_breed_id_checkpoint_v4.pth')
    scheduler.step()
f.close()

run.finish()

