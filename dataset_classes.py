import numpy as np

import warnings
import os
import torch
 
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

warnings.simplefilter('ignore')


class TrainDogBreedDataset(Dataset):
    """
        Dataset class for transforming images into tensors for training
    """
    def __init__(self, root_dir: str, partition: str, labels: np.ndarray, transform: list=[], training_data=True):
        """
            params:
                root_dir: is the root directory containing individual image names(not subdirectories)
                labels: numpy 2D array containing one-hot encoded image labels and ech 1D indice corresponds to 
                the indice or position of target image in the root directory
                transform: list of different transformation/data argumentations to be applied on the images
        """
        
        self.root_dir = root_dir
        self.transforms = transforms.Compose(transform)
        self.images = []  
        
        # reading images from the directory and convert them into PIL images
        image_filenames = os.listdir(self.root_dir)
        num_of_training_samples = int(0.7 * len(image_filenames))
        if training_data:
            image_filenames = image_filenames[: num_of_training_samples]
            self.labels = labels[: num_of_training_samples]
        else:
            image_filenames = image_filenames[num_of_training_samples: ]
            self.labels = labels[num_of_training_samples: ]
                
        for image_filename in image_filenames:
            image = Image.open(os.path.join(self.root_dir, image_filename))
            image = self.transforms(image)
            self.images.append(image)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def denormalize(self, mean, std):
        denormalized_images = []
        denormalize_ = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],std=[1 / s for s in std])
        for index, image in enumerate(self.images):
            denormalized_images.append(denormalize_(image))
        return denormalized_images


class TestDogBreedDataset(Dataset):
    """
        Dataset class for transforming images into tensors for testing
    """
    def __init__(self, root_dir: str, partition: str, transform:list=[]):
        """
            params:
                root_dir: is the root directory containing individual image names(not subdirectories)
                transform: list of different transformation/data argumentations to be applied on the images
        """
        self.root_dir = root_dir
        self.transforms = transforms.Compose(transform)
        self.images = []

        # reading images from the directory and convert them into PIL images
        image_filenames = os.listdir(self.root_dir)
        for image_filename in image_filenames:
            image = Image.open(os.path.join(self.root_dir, image_filename))
            image = self.transforms(image)
            self.images.append(image)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index]
    
