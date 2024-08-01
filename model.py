#git clone https://github.com/Moshtaf/shift-invariant-unet

import torch
import torch.nn as nn
import torch.nn.functional as F

device= 'cuda' if torch.cuda.is_available() else 'cpu'

class Discriminator(nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            self.discriminator_block(in_channels, 64, 2),
            self.discriminator_block(64, 128, 2),
            self.discriminator_block(128, 256, 2),
            self.discriminator_block(256, 512),
            self.discriminator_block(512, 1024),
            self.discriminator_block(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

    def discriminator_block(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
      return nn.Sequential(
          nn.MaxPool2d(kernel_size=2, stride=1),
          BlurPool(in_channels, filt_size=1, stride=2),
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
          nn.BatchNorm2d(num_features=out_channels),
          nn.LeakyReLU(0.2, inplace=True) )