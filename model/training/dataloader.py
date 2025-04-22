# Authors: Elliot Weiner
# Date: 2025-4-20
# Description: 
#   This script loads dataset from all locations, checking values automatically. Only Requires max value found in video samples (I.e. output_X)
#
# Sample Usage: python3 dataloader.py


import torch
import numpy as np
import torchvision.transforms as transforms

# File System Structure
# dataset
#   images1
#     FILE_NAME
#       FILE_NAME_XXXX.npy
#   images2
#     FILE_NAME
#       FILE_NAME_XXXX.npy
#   images3
#     FILE_NAME
#       FILE_NAME_XXXX.npy
#   images4
#     FILE_NAME
#       FILE_NAME_XXXX.npy
#   labels
#     FILE_NAME_right.csv
#     FILE_NAME_left.csv

class DDRDataset(torch.utils.data.Dataset):
    def __init__(self, file_name_orig, dataroot, MAX, transform=None):
        '''
        Initialize the dataset.
        '''
        self.transform = transform
        self.root = dataroot
        self.num_files = 0

        ######################################################################
        # INIT
        ######################################################################


        # left and right foot csv's
        self.labels_left = [] # 5 combos 0-4
        self.labels_right = [] # 5 combos 0-4
        # all camera angles
        self.cam1_data = []
        self.cam2_data = []
        self.cam3_data = []
        self.cam4_data = []


        ######################################################################
        # For Each Sample
        ######################################################################


        for sample in range(MAX):
          # sample int number to 5 integers
          sample_num = str(sample).zfill(5)

          file_name = file_name_orig + sample_num

          # count number of lines in csv
          cnt = 0

          # see if sample exists. if it doesnt, slip
          try:
            open(dataroot + "labels/" + file_name + "_left.csv")
          except:
            continue
          
          self.num_files += 1


          ######################################################################
          # LABELS
          ######################################################################


          # read each line of csv - left
          with open(dataroot + "labels/" + file_name + "_left.csv") as f:
              for line in f:
                  # data verification
                  if line == "" or line == "\n":
                      continue

                  self.labels_left.append(int(float(line.strip())))

                  # get number of data samples
                  cnt += 1

          # read each line of csv - left
          with open(dataroot + "labels/" + file_name + "_right.csv") as f:
              for line in f:
                  # data verification
                  if line == "" or line == "\n":
                      continue

                  self.labels_right.append(int(float(line.strip())))


          ######################################################################
          # DATA
          ######################################################################

          # for each sample
          for i in range(cnt-1):
            num = str(i).zfill(5)

            name = file_name + "/" + file_name + "_" + num + ".npy"
            #load in npy
            self.cam1_data.append(np.load(dataroot + "images1/" + name))
            # self.cam2_data.append(np.load(dataroot + "images2/" + name))
            self.cam3_data.append(np.load(dataroot + "images3/" + name))
            # self.cam4_data.append(np.load(dataroot + "images4/" + name))

        #   print(len(self.labels_left), len(self.labels_right), len(self.cam1_data), len(self.cam3_data))

    def __len__(self):
        '''Return length of the dataset.'''
        return len(self.cam1_data)

    # change
    def __getitem__(self, index):
        '''
        Return the ((image1, image2, image3, image4), (label_left, left_right)) tuple.
        This function gets called when you index the dataset.
        '''
        # print("cam1......", len(self.cam1_data), "  index: ", index)
        # fix path separators for the current OS (replace backslashes with forward slashes)
        image1 = torch.from_numpy(self.cam1_data[index]).float()
        # image2 = torch.from_numpy(self.cam2_data[index]).float()
        image3 = torch.from_numpy(self.cam3_data[index]).float()
        # image4 = torch.from_numpy(self.cam4_data[index]).float()
        label_left = self.labels_left[index]
        label_right = self.labels_right[index]

        # convert labels to one hot encoding
        target_left = torch.zeros(5)
        target_left[label_left] = 1.0

        target_right = torch.zeros(5)
        target_right[label_right] = 1.0

        # apply transformations if any
        image1 = image1.permute(0, 3, 1, 2)
        image3 = image3.permute(0, 3, 1, 2)
        if self.transform:
            image1 = self.transform(image1)
            # image2 = self.transform(image2)
            image3 = self.transform(image3)
            # image4 = self.transform(image4)

        # images = [image1, image2, image3, image4]
        images = [image1, image3]

        targets = [target_left, target_right]

        return images, targets

def getloaders(MAX, batch_size=8):
    ######################################################################
    # INIT
    ######################################################################

    # take max file number
    MAX += 1

    FILE_NAME = "output_"
    dataroot = "../../dataset/"

    # define transforms
    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Resize(size=(156, 156)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.RandomRotation(degrees=10)
    ])


    ######################################################################
    # CREATE DATASET
    ######################################################################


    dataset = DDRDataset(FILE_NAME, dataroot, MAX, transform=transform)


    # split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # data split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # samplers to randomize order of data
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.RandomSampler(test_dataset)


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                            shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                            shuffle=False, num_workers=1)
    
    return trainloader, testloader, dataset, train_dataset, test_dataset
    
if __name__ == "__main__":
    ######################################################################
    # get loaders and dataset
    ######################################################################

    # take max file number
    MAX = 14 # CHANGE TO MAX NUMBER VALUE IN SET
    train_loader, test_loader, dataset, train_dataset, test_dataset = getloaders(MAX)


    ######################################################################
    # TEST SIZES OF DATA
    ######################################################################


    print("Train size: ", len(train_dataset))
    print("Test size: ", len(test_dataset))

    print("size: ", len(dataset))
    print("num files: ", dataset.num_files)


