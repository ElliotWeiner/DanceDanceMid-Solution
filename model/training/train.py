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

import torch

def calculate_accuracy(model, testloader, device=None):

    correct_predictions_l = 0
    correct_predictions_r = 0
    total_samples = 0
    model.eval()
    # disable gradient calc
    with torch.no_grad():
        # iterate over testloader
        for batch_idx, batch in enumerate(testloader):
            # move data to the specified device
            
            batch_in_image_1, batch_in_image_3, batch_gt_left, batch_gt_right = batch[0][0].to(device), batch[0][1].to(device), batch[1][0].to(device), batch[1][1].to(device)

            # model predictions
            batch_out_left, batch_out_right = model(batch_in_image_1, batch_in_image_3)

            batch_out_left = torch.nn.functional.softmax(batch_out_left.squeeze(0), dim=1)
            batch_out_right = torch.nn.functional.softmax(batch_out_right.squeeze(0), dim=1)
            
            # max index
            _, pr_l = torch.max(batch_out_left.data, 1)
            _, pr_r = torch.max(batch_out_right.data, 1)
            _, gt_l = torch.max(batch_gt_left.data, 1)
            _, gt_r = torch.max(batch_gt_right.data, 1)
            
            print(pr_l, gt_l, pr_r, gt_r)
            
            #print(predicted_l.shape, batch_gt_left.shape)

            # update total samples
            total_samples += batch_gt_right.size(0)

            # Compare predicted classes with true labels
            correct_predictions_l += (pr_l == gt_l).sum().item()
            correct_predictions_r += (pr_r == gt_r).sum().item()
            
            #print(correct_predictions_l, correct_predictions_r)


    # calculate accuracy
    accuracy_l = 100 * correct_predictions_l / total_samples if total_samples > 0 else 0.0
    accuracy_r = 100 * correct_predictions_r / total_samples if total_samples > 0 else 0.0
    #accuracy_t = 100 * (correct_predictions_l + correct_predictions_r) / total_samples if total_samples > 0 and total_samples > 0 else 0.0

	
    return accuracy_l, accuracy_r#, accuracy_t



def train_the_feet(save_path, lr, num_povs, MAX):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """

    ######################################################################
    # INIT
    ######################################################################

    print("Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feet_net = FeetNet(num_povs).to(device)
    #optimizer = torch.optim.Adam(feet_net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(feet_net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=300)
    criterion = torch.nn.CrossEntropyLoss()

    nr_epochs = 500
    batch_size = 16
    start_time = time.time()

    all_data = getloaders(MAX, batch_size)
    train_loader = all_data[0]
    test_loader = all_data[1]


    ######################################################################
    # TRAINING LOOP - FOR EACH EPOCH
    ######################################################################
    print_interval = 5#int(len(train_loader)/batch_size)//2
    losses = []
    l_losses = []
    r_losses = []
    acc_l = []
    acc_r = []
    for epoch in range(nr_epochs):
        total_loss_l = 0
        total_loss_r = 0
        total_loss = 0
        batch_in = []
        batch_gt = []


        ######################################################################
        # FOR EACH BATCH
        ######################################################################

        feet_net.train()
        for batch_idx, batch in enumerate(train_loader):
            # print(batch)
            batch_in_image_1, batch_in_image_3, batch_gt_left, batch_gt_right = batch[0][0].to(device), batch[0][1].to(device), batch[1][0].to(device), batch[1][1].to(device)

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

            if batch_idx % print_interval == print_interval - 1:
                last_loss = total_loss / print_interval # loss per batch
                last_ll = total_loss_l / print_interval
                last_lr = total_loss_r / print_interval

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
        #scheduler.step()
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tloss_l: %.6f\tloss_r: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, last_loss, last_ll, last_lr, lrBefore, lrAfter, time_left))
        
        if epoch % 100 == 0 and epoch != 0:
            print("Saving Model Checkpoint")
            torch.save(feet_net, save_path + str(epoch) + "-feet_net.pth")

        
        # check accuracy every 5 epochs
        #feet_net.eval()
        if epoch % 5 == 0 and epoch != 0:
            test_accuracy_l, test_accuracy_r = calculate_accuracy(feet_net, test_loader, device)
            acc_l.append(test_accuracy_l)
            acc_r.append(test_accuracy_r)
            print("Test Accuracy L: %.6f" % (test_accuracy_l))
            print("Test Accuracy R: %.6f" % (test_accuracy_r))

        l_losses.append(last_ll)
        r_losses.append(last_lr)
        losses.append(last_loss)
        if last_loss <= 0.02 or (last_ll <= 0.02 and last_lr <= 0.02):
            test_accuracy_l, test_accuracy_r = calculate_accuracy(feet_net, test_loader, device)
            acc_l.append(test_accuracy_l)
            acc_r.append(test_accuracy_r)
            print("Test Accuracy L: %.6f" % (test_accuracy_l))
            print("Test Accuracy R: %.6f" % (test_accuracy_r))
            break

    
    ######################################################################
    # POST-TRAINING STATISTICS, PLOT & SAVE
    ######################################################################


    print("Last loss: ", last_loss)
    print("Last_loss_l: ", last_ll)
    print("Last_loss_r: ", last_lr)

    
    cpuLoss = [loss.cpu().detach().float() for loss in losses]
    cpuLossL = [loss.cpu().detach().float() for loss in l_losses]
    cpuLossR = [loss.cpu().detach().float() for loss in r_losses]

    
    torch.save(feet_net, save_path + "final_feet_net.pth")
    epochs = list(len(cpuLoss))
    print(epochs);
    print(cpuLoss);
    tl = plt.plot(epochs, cpuLoss, label="Total Loss")
    ll = plt.plot(epochs, cpuLossL, label="L Loss")
    rl = plt.plot(epochs, cpuLossR, label="R Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.title("Epoch vs Training Loss")
    plt.legend([tl, ll, rl])
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
 
