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

    feet_net = FeetNet(num_povs).to(gpu)
    optimizer = torch.optim.Adam(feet_net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(feet_net.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=50)
    criterion = torch.nn.CrossEntropyLoss()

    nr_epochs = 100
    batch_size = 16
    start_time = time.time()

    all_data = getloaders(MAX, batch_size)
    train_loader = all_data[0]
    test_loader = all_data[1]


    ######################################################################
    # TRAINING LOOP - FOR EACH EPOCH
    ######################################################################


    losses = []
    for epoch in range(nr_epochs):
        total_loss_l = 0
        total_loss_r = 0
        total_loss = 0
        batch_in = []
        batch_gt = []


        ######################################################################
        # FOR EACH BATCH
        ######################################################################


        for batch_idx, batch in enumerate(train_loader):
            # print(batch)
            batch_in_image_1, batch_in_image_3, batch_gt_left, batch_gt_right = batch[0][0].to(gpu), batch[0][1].to(gpu), batch[1][0].to(gpu), batch[1][1].to(gpu)

            optimizer.zero_grad()

            # print(batch_in_image_1.shape)
            batch_out_left, batch_out_right = feet_net(batch_in_image_1, batch_in_image_3)

            # print(batch_out_left.shape)
            # print(batch_gt_left.shape)
            # print(batch_gt_left, batch_out_left)
            batch_out_left = batch_out_left.squeeze(0)
            batch_out_right = batch_out_right.squeeze(0)
            loss_l = criterion(batch_out_left, batch_gt_left)
            loss_r = criterion(batch_out_right, batch_gt_right)
            loss = loss_l + loss_r

            loss.backward()
            optimizer.step()
            total_loss_l += loss_l
            total_loss_r += loss_r
            total_loss += loss
            # running_loss += loss

            if batch_idx % 10 == 9:
                last_loss = total_loss / 10 # loss per batch
                last_ll = total_loss_l / 10
                last_lr = total_loss_r / 10

                print('batch {} loss: {}, ll: {}, lr: {}'.format(batch_idx + 1, last_loss, last_ll, last_lr))
                total_loss = 0.
                total_loss_l = 0.
                total_loss_r = 0.
        

        ######################################################################
        # STATISTICS
        ######################################################################


        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        lrBefore = optimizer.param_groups[0]["lr"]
        scheduler.step()
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tloss_l: %.6f\tloss_r: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, last_loss, last_ll, last_lr, lrBefore, lrAfter, time_left))
        
        if epoch % 10 == 0 and epoch != 0:
            print("Saving Model Checkpoint")
            torch.save(feet_net, save_path + str(epoch) + "-feet_net.pth")


        losses.append(last_loss)
        if last_loss <= 0.01:
            break

    
    ######################################################################
    # POST-TRAINING STATISTICS, PLOT & SAVE
    ######################################################################


    print("Training finished in %.2fm" % (time.time() - start_time)/60.0)
    print("Total loss: ", total_loss)
    print("Total_loss_l: ", total_loss_l)
    print("Total_loss_l: ", total_loss_r)

    
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
# TRAINING LOOP
######################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feet Net Traininging')
    parser.add_argument('-l', '--learning_rate', default="0.00001", type=float, help='learning rate')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    parser.add_argument('-m', '--max_file', default="10000", type=int, help='number of max file name')
    args = parser.parse_args()
    
    train_the_feet(args.save_path, args.learning_rate, 2, args.max_file)
 
