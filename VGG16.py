import torch
import torch.nn as tnn
import numpy as np

class VGG16(tnn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = tnn.Sequential(
            # layer 1 (conv3-64)
            tnn.Conv2d(1, 64, kernel_size=3, padding=2),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            # layer 1 (conv3-64)
            tnn.Conv2d(64, 64, kernel_size=3, padding=2),
            tnn.BatchNorm2d(64),
            tnn.ReLU(), 
            # layer 1 pooling
            tnn.MaxPool2d(kernel_size=2, stride=2)         
        )
        self.layer2 = tnn.Sequential(
            # layer 2 (conv3-128)
            tnn.Conv2d(64, 128, kernel_size=3, padding=2),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            # layer 2 (conv3-128)
            tnn.Conv2d(128, 128, kernel_size=3, padding=2),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            # layer 2 pooling
            tnn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = tnn.Sequential(
            # layer 3 (conv3-256)
            tnn.Conv2d(128, 256, kernel_size=3, padding=2),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            # layer 3 (conv3-256)
            tnn.Conv2d(256, 256, kernel_size=3, padding=2),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),
            # layer 3 pooling
            tnn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = tnn.Sequential(
            # layer 4 (conv3-512)
            tnn.Conv2d(256, 512, kernel_size=3, padding=2),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            # layer 4 (conv3-512)
            tnn.Conv2d(512, 512, kernel_size=3, padding=2),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            # layer 4 pooling
            tnn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = tnn.Sequential(
            # layer 5 (conv3-512)
            tnn.Conv2d(512, 512, kernel_size=3, padding=2),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            # layer 5 (conv3-512)
            tnn.Conv2d(512, 512, kernel_size=3, padding=2),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),
            # layer 5 pooling
            tnn.MaxPool2d(kernel_size=2, stride=2)           
        )
        self.layer6 = tnn.Sequential(
            tnn.Linear(22528, 4096),
            tnn.BatchNorm1d(4096),
            tnn.ReLU()
        )
        self.layer7 = tnn.Sequential(
            tnn.Linear(4096, 4096),
            tnn.BatchNorm1d(4096),
            tnn.ReLU()
        )
        self.layer8 = tnn.Sequential(
            tnn.Linear(4096, 228),
            tnn.BatchNorm1d(228),
            # tnn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        out = self.layer5(out)
        vgg16_features = out.view(out.size(0), -1)
        out = self.layer6(vgg16_features)
        out = self.layer7(out)
        out = self.layer8(out)
        out = torch.unsqueeze(out, 2)
        # print(out.shape)
        return vgg16_features, out
