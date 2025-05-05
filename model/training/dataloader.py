# Authors: Elliot Weiner
# Date: 2025-4-20
# Description: 
#   This script loads dataset from all locations, checking values automatically. Only Requires max value found in video samples (I.e. output_X)
#
# Sample Usage: python3 dataloader.py

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
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
          for i in range(cnt):
            num = str(i).zfill(4)

            name = file_name + "/" + file_name + "_" + num + ".npy"
            #load in npy
            # self.cam1_data.append(np.load(dataroot + "images1/" + name))
            self.cam2_data.append(np.load(dataroot + "images1/" + name))
            self.cam2_data[-1][0, :, :, :] -= self.cam2_data[-1][1, :, :, :]
            self.cam2_data[-1][1, :, :, :] -= self.cam2_data[-1][2, :, :, :]
            # self.cam3_data.append(np.load(dataroot + "images3/" + name))
            self.cam4_data.append(np.load(dataroot + "images2/" + name))
            self.cam4_data[-1][0, :, :, :] -= self.cam4_data[-1][1, :, :, :]
            self.cam4_data[-1][1, :, :, :] -= self.cam4_data[-1][2, :, :, :]

        print(len(self.labels_left), len(self.labels_right))#, len(self.cam2_data), len(self.cam4_data))

		#self.labels_left = self.labels_left[:10]


        # random.seed(seed)
        l_targets = np.asarray(self.labels_left)
        r_targets = np.asarray(self.labels_right)
        cam2 = np.asarray(self.cam2_data)
        cam4 = np.asarray(self.cam4_data)
        class_indices = np.where((l_targets == 4.0) & (r_targets == 4.0))[0]
        other_indices = np.where((l_targets != 4.0) | (r_targets != 4.0))[0]


        #num_to_keep = int(len(class_indices))#* 0.75)
        num_to_keep = 1750
        keep_indices = random.sample(list(class_indices), num_to_keep)
        #keep_indices = class_indices

        
        #other_indices = random.sample(list(other_indices), 400)

        # Keep all other class indices
        final_indices = np.concatenate([keep_indices, other_indices])
        np.random.shuffle(final_indices)
        self.labels_left_bal = l_targets[final_indices]#torch.utils.data.Subset(l_targets, final_indices)
        self.labels_righ_bal = r_targets[final_indices]#torch.utils.data.Subset(r_targets, final_indices)
        # self.cam2_data_bal = cam2[final_indices]
        # self.cam4_data_bal = cam4[final_indices]

        # print("Left Before: ", len(self.labels_left))
        # print("Left After: ", self.labels_left_bal.shape)
        print(len(self.labels_left_bal), len(self.labels_righ_bal))#, len(self.cam2_data_bal), len(self.cam4_data_bal))
        
        self.actual_labels = []
																																													#[U, D, L, R, UD, UL, UR, LR, RD, DL, N]
																																													#[0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10]
		
        mapping = {(0, 0): 0, (0, 4): 0, (4, 0): 0, (1, 1): 1, (4, 1): 1, (1, 4): 1, (2, 2): 2, (2, 4): 2, (4, 2): 2, (3, 3): 3, (3, 4): 3, (4, 3): 3, (4, 4): 10, (0, 1): 4, (1, 0): 4, (0, 2): 5, (2, 0): 5, (0, 3): 6, (3, 0): 6, (2, 3): 7, (3, 2): 7, (1, 3): 8, (3, 1): 8, (1, 2): 9, (2, 1): 9}
        
        dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
        for l, r in zip(self.labels_left_bal, self.labels_righ_bal):
            if mapping[(l, r)] == 10:
                self.actual_labels.append(4)
                dist[4] += 1
            elif mapping[(l, r)] < 4:
                self.actual_labels.append(mapping[(l,r)])
                dist[mapping[(l,r)]] += 1
            else:
                pass

        print(dist)
			


    def __len__(self):
        '''Return length of the dataset.'''
        return len(self.actual_labels)

    # change
    def __getitem__(self, index):
        '''
        Return the ((image1, image2, image3, image4), (label_left, left_right)) tuple.
        This function gets called when you index the dataset.
        '''
        # print("cam1......", len(self.cam1_data), "  index: ", index)
        # fix path separators for the current OS (replace backslashes with forward slashes)
        # image1 = torch.from_numpy(self.cam1_data[index]).float()
        #test = self.cam2_data[index]
        #test = test[-1, :, :, :]
        #print(test.shape)
        #plt.imshow(test)
        #plt.show()
        image2 = torch.from_numpy(self.cam2_data[index]).float()
        # image3 = torch.from_numpy(self.cam3_data[index]).float()
        image4 = torch.from_numpy(self.cam4_data[index]).float()
        #label_left = self.labels_left_bal[index]
        #label_right = self.labels_righ_bal[index]
        #image2 = image2[-1, :, :, :].squeeze(0)
        #print(image2)
        #print(image2.shape)
        #image4 = image4[-1, :, :, :].squeeze(0)
        # convert labels to one hot encoding
        #target_left = torch.zeros(5)
        #target_left[label_left] = 1.0

        #target_right = torch.zeros(5)
        #target_right[label_right] = 1.0
        target = torch.zeros(5)
        target[self.actual_labels[index]] = 1.0
        
        #plt.imshow(image2 * 255.0)
        #plt.show()
        # apply transformations if any
        #image2 = image2.permute(2, 0, 1)

        #cthw
        #thwc
        
        #image2 = image2.permute(0, 3, 1, 2)
        #image4 = image4.permute(2, 0, 1)
        #image4 = image4.permute(0, 3, 1, 2)

        image2 = image2.permute(1, 0, 2, 3)
        image4 = image4.permute(1, 0, 2, 3)
        
        

        if self.transform:
            # image1 = self.transform(image1)
            image2 = self.transform(image2)
            # image3 = self.transform(image3)
            image4 = self.transform(image4)

        # images = [image1, image2, image3, image4]
        # print(image2.shape, image4.shape, target_left, target_right)
        
        image2 = image2.permute(1, 0, 2, 3)
        image4 = image4.permute(1, 0, 2, 3)
        #print(image2.shape)
        
        #test = image2.permute(1, 2, 0)
        #plt.imshow(test)
        #plt.show()
        images = [image2, image4]

        #targets = [target_left, target_right]

        return images, target#targets

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
        #transforms.Resize(size=(112, 112)),
        #transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Normalize((0.4316, 0.3945, 0.3765), (0.228, 0.2215, 0.2170)),
        transforms.RandomRotation(degrees=10)
    ])
    
    transform_test = transforms.Compose([
        #transforms.Resize(size=(112, 112)),
        transforms.Normalize((0.4316, 0.3945, 0.3765), (0.228, 0.2215, 0.2170))
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


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
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


