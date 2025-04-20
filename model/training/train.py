# Authors: Arthur Wang
# Date: 2025-4-20
# Description: 
#   This script runs the main training loop for the model
#
# Sample Usage: python3 train.py --learning_rate 0.00001 --save_path ./

import time
import random
import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from FeetNet import FeetNet
from dataloader import DDRDataset, getloaders

def train_the_feet(save_path, lr, num_povs, MAX):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """

    ######################################################################
    # INIT
    ######################################################################


    gpu = torch.device('cuda')

    feet_net = FeetNet(2).to(gpu)
    optimizer = torch.optim.Adam(feet_net.parameters(), lr=lr)

    nr_epochs = 100
    batch_size = 8
    start_time = time.time()

    all_data = getloaders(MAX, batch_size)
    train_loader = all_data[0]
    test_loader = all_data[1]


    ######################################################################
    # TRAINING LOOP - FOR EACH EPOCH
    ######################################################################


    losses = []
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []


        ######################################################################
        # FOR EACH BATCH
        ######################################################################


        for batch_idx, batch in enumerate(train_loader):
            batch_in_image, batch_gt = batch[0][0].to(gpu), batch[1].to(gpu)

            batch_out = feet_net(batch_in_image[0], batch_in_image[2])
            batch_gt = feet_net.actions_to_classes(batch_gt)

            loss = cross_entropy_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        

        ######################################################################
        # STATISTICS
        ######################################################################


        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        lrBefore = optimizer.param_groups[0]["lr"]
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, total_loss, lrBefore, lrAfter, time_left))

        losses.append(total_loss)
        if total_loss <= 0.01:
            break

    
    ######################################################################
    # POST-TRAINING STATISTICS, PLOT & SAVE
    ######################################################################


    print("Training finished in %.2fm" % (time.time() - start_time)/60)
    print("Total loss: ", total_loss)
    
    cpuLoss = [loss.cpu().detach().float() for loss in losses]

    
    torch.save(feet_net, save_path + "feet_net.pth")
    epochs = list(range(nr_epochs))
    print(epochs)
    print(cpuLoss)
    plt.plot(epochs, cpuLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Training Loss")
    plt.savefig('training_loss_class.png')
    plt.show()


######################################################################
# DEF LOSS
######################################################################


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """

    #loss = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
    batch_gt = torch.tensor(batch_gt).cuda()
    lpred = torch.log(batch_out)
    ytruelogpred = torch.mul(batch_gt, lpred)
    loss_tensor = -torch.mean(torch.sum(ytruelogpred, dim=1))

    return loss_tensor



######################################################################
# TRAINING LOOP
######################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feet Net Traininging')
    parser.add_argument('-l', '--learning_rate', default="0.00001", type=float, help='learning rate')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    parser.add_argument('-m', '--max_file', default="10000", type=int, help='number of max file name')
    args = parser.parse_args()
    
    train_the_feet(args.save_path, args.learning_rate, 2, args.max_file)
 
