import torch
from torch import nn
import torchvision.models as models


# defining model architecture
class DogBreedNet(nn.Module):
    """
        Model architecture for dog breed identification and classification
        The model expects the images of size 400 x 400
    """
    def __init__(self, input_channels, num_classes, dense_dropout_p):
        """
            model instance construct
            params:
                input_channels: the number of channels for images(RGB -> 3 channels, grayscale -> one channel)
                num_classes: the number of classes for which images has to be classified into
                conv_dropout_p: the probability(percentage) of neurons to be dropped in the convolutional layers during training
                dense_dropout_p: the probability(percentage) of neurons to be dropped during in training in fully connected network(MLP)
        """
        super().__init__()
        self.block_1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.BatchNorm2d(128)
                                  )
        
        
        self.block_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.BatchNorm2d(128)
                                  )
        
        self.block_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2)
                                  )
        
        self.block_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),
                                  nn.ReLU(),
                                  nn.AvgPool2d(kernel_size=2, stride=2)
                                  )
        
        self.flatten_layer = nn.Flatten()
        
        self.mlp = nn.Sequential(nn.Linear(in_features=512 * 10 * 10, out_features=1024),
                              nn.ReLU(),
                              nn.Dropout(dense_dropout_p),
                              
                              nn.Linear(in_features=1024, out_features=2048),
                              nn.ReLU(),
                              nn.Dropout(dense_dropout_p),
                               
                              nn.Linear(in_features=2048, out_features=num_classes)
                              )
        
    def forward(self, X):
        """
            params:
                X: batch of images to be passed into the model
                X has shape of (batch_size, num_channels, height, width)
        """
        out = self.block_1(X)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.flatten_layer(out)  # passing output of cnn layers to flatten layer
        out = self.mlp(out) # passing output of flatten layer to fully connected network
        
        return out

def create_model(num_classes, dense_dropout_p):
    resnet = models.resnet101(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(nn.Linear(in_features=num_features, out_features=2048),
                              nn.ReLU(),
                              nn.Dropout(dense_dropout_p),
                              
                              nn.Linear(in_features=2048, out_features=2048),
                              nn.ReLU(),
                              nn.Dropout(dense_dropout_p),
                               
                              nn.Linear(in_features=2048, out_features=num_classes)
                              )
    return resnet
