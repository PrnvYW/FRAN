import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from skimage import io, transform



import os
import random
import math
import python_utils as utils
import unittest
import logging

from tqdm import tqdm as tqdm

DATADIR='/content/data_final'
LOGFILE = "./age.ai.log"
BATCH_SIZE= 2
logging.basicConfig(level=logging.DEBUG, filename=LOGFILE)



# Some parameters used below are defined here - ideally they come from a config file that can be easily changed
INPUT_CLASSES = [20., 30., 40., 50., 60., 70., 80.]
TARGET_CLASSES = [20., 30., 40., 50., 60., 70., 80.]
IMAGE_SIZE = 1024


class ImagePairsDataset(datasets.DatasetFolder):
    """Returns a pair of images with labels. Assumes directory structure:
        input_image_dir/class1/
        input_image_dir/class2/
        etc
        The same filename must be present in each class directory.
        Pairs consist of corresponding files from different class directories e.g.
        input_image_dir/class1/img1.png and input_image_dir/class2/img1.png are a pair

        Todo: Generalize this later to also return pairs of different file names but from the same directory
    """

    def __init__(self, input_image_dir, loader=imageLoader, transform=None,
                 extensions=[".jpg", ".jpeg", ".png"]):
        """
        Args:
            input_image_dir (string): Directory with images to age
            target_image_dir (string): Directory with corresponding aged images
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        print("Input image dir is: ", input_image_dir, os.path.isdir(input_image_dir))
        assert os.path.isdir(input_image_dir)

        self.input_image_dir = input_image_dir
        self.transform = transform
        self.means = (0.5, 0.5, 0.5)
        self.std_devs = (0.5, 0.5, 0.5)

        super().__init__(input_image_dir, loader=loader, transform=transform, extensions=extensions)

        self.input_classes = INPUT_CLASSES
        self.target_classes = TARGET_CLASSES

        # The same filenames should be in each dir. Count files in any class.
        classdir = self.input_image_dir + os.sep + str(int(self.input_classes[0]))
        self.image_names = []
        for file in os.listdir(classdir):
            path = classdir + os.sep + file
            _, extension = os.path.splitext(file)
            if extension in self.extensions and os.path.isfile(path):
                self.image_names.append(file)
        self.num_images_per_class = len(self.image_names)

        # Todo: Validate dataset - check if any files are missing in any class. Log results.


    def __len__(self):
        num_image_pairs = self.num_images_per_class * len(self.input_classes) * len(self.target_classes)

        return num_image_pairs

    # Returns input image and its target aged image
    def __getitem__(self, idx):
        fileindex, remainder = divmod(idx, len(self.input_classes) * len(self.target_classes))
        input_class_idx, target_class_idx = divmod(remainder, len(self.target_classes))


        filename1 = os.path.join(self.input_image_dir, str(int(self.input_classes[input_class_idx])), self.image_names[fileindex])
        filename2 = os.path.join(self.input_image_dir, str(int(self.target_classes[target_class_idx])), self.image_names[fileindex])

        image1 = plt.imread(filename1)
        if image1 is None:
            raise Exception(
                "Error: Image " + filename1 + " not found. ")

        image2 = plt.imread(filename2)
        if image2 is None:
            raise Exception(
                "Error: Image " + filename2 + " not found. ")
        # Dataset creation should have resized images
        if image1.shape[0:2] != (IMAGE_SIZE, IMAGE_SIZE):
            print(f"{filename1} is not resized!", image1.shape[0:2] )
        assert image1.shape[0:2] == (IMAGE_SIZE, IMAGE_SIZE)
        assert image2.shape[0:2] == (IMAGE_SIZE, IMAGE_SIZE)
        if image2.shape[0:2] != (IMAGE_SIZE, IMAGE_SIZE):
            print(f"{filename2} is not resized!", image2.shape[0:2] )

        sample = {'input_image': image1,
                  'target_image_available': True, 'target_image': image2,
                  'filename': self.image_names[fileindex],
                  'input_ages' : self.input_classes[input_class_idx],
                  'target_ages' : self.target_classes[target_class_idx]
                  }

        if self.transform:
            sample = self.transform(sample)

        #sample['input_ages'] = self.input_classes[input_class_idx]
        #sample['target_ages'] = self.target_classes[target_class_idx]

        return sample

    def unnormalize_image(self, image):
        for channel in range(image.shape[0]):
            image[channel] = image[channel] * self.std_devs[channel] + self.means[channel]
        return image

    class Channel(object):
      """Adding 2 extra channels"""

      def __call__(self, sample):
        chn4=torch.ones(1, sample['input_image'].shape[1], sample['input_image'].shape[2])*sample['input_ages']/100
        chn5=torch.ones(1, sample['input_image'].shape[1], sample['input_image'].shape[2])*sample['target_ages']/100
        chn4=chn4.to(device)
        chn5=chn5.to(device)
        sample['input_image']=torch.concat((sample['input_image'], chn4, chn5), 0)
        return sample

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W

            # Pytorch for Mac GPU doesnt support float 64 - cast to float32
            sample['input_image'] = TF.to_tensor(sample['input_image']).float().to(device)
            sample['target_image'] = TF.to_tensor(sample['target_image']).float().to(device)

            return sample

    class Normalize(object):
       """Normalize input and target. Also scales from -1 to 1 (pixel = (pixel - 0.5)/0.5"""

       def __call__(self, sample):
           sample['input_image'] = TF.normalize(sample['input_image'], (0.5, 0.5, 0.5, 0, 0), (0.5, 0.5, 0.5, 1, 1))
           sample['target_image'] = TF.normalize(sample['target_image'], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

           return sample

    class RandomCrop(object):
      """Sampling Random Crops of size 512"""

      def __init__(self, size=(512, 512)):
        self.size = size
        self.crop_transform = transforms.RandomCrop(size)

      def __call__(self, sample):
        # Generate a random crop position
        i, j, h, w = transforms.RandomCrop.get_params(sample['input_image'], output_size=self.size)

        # Apply the crop to both images
        sample['input_image'] = transforms.functional.crop(sample['input_image'], i, j, h, w)
        sample['target_image'] = transforms.functional.crop(sample['target_image'], i, j, h, w)

        return sample


    @staticmethod
    def get_transforms():
        transform = transforms.Compose([
            ImagePairsDataset.ToTensor(),
            ImagePairsDataset.Channel(),
            ImagePairsDataset.RandomCrop(),
            ImagePairsDataset.Normalize(),

        ])
        return transform



def get_dataloaders(dataset_dir=DATADIR, batch_size=BATCH_SIZE, shuffle=True):

    image_datasets = {x: ImagePairsDataset(os.path.join(dataset_dir, x), transform=ImagePairsDataset.get_transforms()) \
                      for x in ['train','val']}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                       shuffle=False if x != 'train' else shuffle, num_workers=0) for x
                        in ['train', 'val']}
    return dataloaders_dict





def testdataloader():
    loaders = get_dataloaders(shuffle=False)
    valloader = loaders['val']
