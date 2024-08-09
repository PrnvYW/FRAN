import numpy as np
import pandas as pd
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

device= 'cuda' if torch.cuda.is_available() else 'cpu'

#loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
w1=1
w_ad=0.05
w_p=1


def get_discriminator_loss(generated_samples, discriminator, criterion, inp, target, target_age):

    #Appending Constant Channel of target age to discriminator input
    chn4=torch.ones(generated_samples.shape[0], 1, generated_samples.shape[2], generated_samples.shape[3])*target_age[0]
    chn4=chn4.float().to(device)
    chn4=chn4/100
    generated_samples=generated_samples+inp[:, 0:3, :, :]
    generated_samples=torch.cat((generated_samples, chn4), 1)

    target=torch.cat((target, chn4), 1)

    discriminator_fake_pred = discriminator(generated_samples)
    discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))
    discriminator_real_pred = discriminator(target)
    discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))
    discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

    return discriminator_loss

def get_generator_loss(generated_samples, discriminator, criterion, inp, target_age):


    #Appending Constant Channel of target age to discriminator input
    chn4=torch.ones(generated_samples.shape[0], 1, generated_samples.shape[2], generated_samples.shape[3])*target_age[0]
    chn4=chn4.float().to(device)
    chn4=chn4/100
    generated_samples=generated_samples+inp[:, 0:3, :, :]
    generated_samples=torch.cat((generated_samples, chn4), 1)

    discriminator_fake_pred = discriminator(generated_samples.detach())
    generator_loss = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))

    return generator_loss



def Loss(generated ,discriminator, inp, target, target_age):

  L_ad=get_generator_loss(generated, discriminator, nn.BCEWithLogitsLoss(), inp, target_age)

  generated=generated+inp[:, 0:3, :, :]
  L_p=loss_fn_vgg(generated, target)

  L=nn.L1Loss().to(device)
  L_1=L(generated, target)
  loss= (w1*L_1+w_p*L_p+w_ad*L_ad).sum()
  return loss, L_1, L_p.sum(), L_ad
