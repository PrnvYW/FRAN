#%cd /content/shift-invariant-unet
from pbpunet.pbpunet_model import PBPUNet
lpf_sizd=[3, 3, 3, 3]

device= 'cuda' if torch.cuda.is_available() else 'cpu'
generator=PBPUNet(n_channels=5, n_classes=3, lpf_size=lpf_size).to(device)

def preprocess(path, in_age, out_age):
  input=cv2.imread(path)
  input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)


  #Converting to tensor and making dim as required by the model
  input = TF.to_tensor(input).float()
  input=input.to(device)

  print(input.shape)

  #Concatenating age channels
  chn4=torch.ones(1, input.shape[1], input.shape[2])*in_age/100
  chn5=torch.ones(1, input.shape[1], input.shape[2])*out_age/100
  chn4=chn4.to(device)
  chn5=chn5.to(device)
  input=torch.concat((input, chn4, chn5), 0)

  print(input.shape)
  input=input[None, :, :, :]

  input[0] = TF.normalize(input[0], (0.5, 0.5, 0.5, 0, 0), (0.5, 0.5, 0.5, 1, 1))

  return input


def inference(path, in_age, out_age):
  #Reading input
  input=preprocess(path, in_age, out_age)

  with torch.no_grad():
    target=generator(input).to(device)
  
  target=torch.transpose(target, 1, 2)
  target=torch.transpose(target, 2, 3)
  input=torch.transpose(input, 1, 2)
  input=torch.transpose(input, 2, 3)

  target=(target.to('cpu').numpy())
  input=(input.to('cpu').numpy())

  input[:, :, :, 0:3]=input[:, :, :, 0:3]+target

  #Denormalising
  for channel in range(target.shape[3]):
    input[0, :, :, channel] = input[0, :, :, channel] * 0.5+ 0.5


  #plt.imshow(target[0])

  return input[0, :, :, 0:3]
