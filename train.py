import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm

device= 'cuda' if torch.cuda.is_available() else 'cpu'

from losses import get_discriminator_loss, Loss

from dataloaders import ImagePairsDataset, get_dataloaders
from pbpunet.pbpunet_model import PBPUNet
from model import Discriminator
lpf_size=[3, 3, 3, 3]

PATH='/content/data_final/train' 
dataset=ImagePairsDataset(PATH)
data_loaders= get_dataloaders(shuffle=False)
train_loader=data_loaders['train']
val_loader=data_loaders['val']

# Generator
generator=PBPUNet(n_channels=5, n_classes=3, lpf_size=lpf_size).to(device)
generator_optim = torch.optim.Adam(generator.parameters(), lr=0.0001)

# Discriminator
discriminator=Discriminator(in_channels=4).to(device)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

criterion = nn.BCEWithLogitsLoss()

#Training Loop
mean_generator_loss = 0
mean_discriminator_loss = 0
n_epochs=100

"""state=torch.load(MODEL_PATH)
generator.load_state_dict(state['gen_state_dict'])
generator_optim.load_state_dict(state['gen_optim_state_dict'])
n_epochs= n_epochs- state['epoch']
discriminator.load_state_dict(state['dis_state_dict'])
discriminator_optim.load_state_dict(state['dis_optim_state_dict'])"""

for epoch in range(n_epochs):
    for samples in tqdm(train_loader):

        generated=generator(samples['input_image']).to(device)
        # train discriminator
        discriminator_optim.zero_grad()
        discriminator_loss = get_discriminator_loss(generated, discriminator, criterion, samples['input_image'], samples['target_image'], samples['target_ages'])
        discriminator_loss.backward(retain_graph=True)
        discriminator_optim.step()
        mean_discriminator_loss += discriminator_loss.item()/len(train_loader)

        # train generator
        generator_optim.zero_grad()
        generator_loss, L_1, L_p, L_ad = Loss(generated, discriminator, samples['input_image'], samples['target_image'], samples['target_ages'])
        generator_loss.backward()
        generator_optim.step()
        mean_generator_loss += generator_loss.item()/len(train_loader)

        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1}: Ad loss: {L_ad}, Lpipis: {L_p}, L_1: {L_1}, discriminator loss: {mean_discriminator_loss}")
    if (epoch+1)%10 == 0:
      MODEL_PATH = "model_"+str(epoch+1)+".pt"
      torch.save({
                  'epoch': epoch,
                  'gen_state_dict': generator.state_dict(),
                  'dis_state_dict': discriminator.state_dict(),
                  'gen_optim_state_dict': generator_optim.state_dict(),
                  'dis_optim_state_dict': discriminator_optim.state_dict(),
                  'loss': mean_generator_loss+mean_discriminator_loss,
                  }, MODEL_PATH)
    mean_generator_loss = 0
    mean_discriminator_loss = 0
