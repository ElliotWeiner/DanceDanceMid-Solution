# Authors: Arthur Wang and Elliot Weiner
# Date: 2025-4-6
# Description: 
#   This script chunks a video into smaller segments (gifs)
#   and saves them as numpy arrays for annotation. It then
#   opens a custom annotation GUI to annotate the gifs.
#
# Sample Usage: python3 annotation_chunking.py test_output1.mp4
#   - add a file name as seen in the images1 dataset pertaining to the specific video which will be annotated
#   - repeat data must be first deleted

import cv2
import os
import numpy as np
import argparse
import time
import torch
from torchvision.transforms import transforms


#######################################################################################################
#######################################################################################################
#
#           Data Chunker
#
#######################################################################################################
#######################################################################################################


#######################################################################
# Initialization
#######################################################################

parser = argparse.ArgumentParser(
                    prog='Video Annotato',
                    description='Chunks video and annotates',
                    epilog='Help yourself lol')

parser.add_argument('image')           # positional argument
args = parser.parse_args()


# path declaration
path = "D:/Mixed_crop/" + args.image


folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
folders.sort()

print(folders)

images = []


def mouse_callback(event, mouse_x, mouse_y, flags, param):
    global x, y
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        x, y = mouse_x, mouse_y
        print(f"Clicked at ({x}, {y})")

gui = cv2.namedWindow("GUI", cv2.WINDOW_NORMAL)

disp = np.zeros((600, 800, 3), dtype=np.uint8)
cv2.imshow('GUI', disp)

resize = transforms.Resize((112, 112))



x, y = 0, 0
w = 256
h = 256

cv2.setMouseCallback('GUI', mouse_callback)

start = False

for video_output in folders:
    if start == False:
        if video_output == "output_00009":
            start = True
            print("Starting: ", video_output)
        else:
            print("Skipping: ", video_output)
            continue
        
    image_paths = [name for name in os.listdir(path + "/" + video_output) if name[0] != "."]
    # break


    image_paths.sort()
    print(image_paths)

    cropped = []


    for image_path in image_paths:
        print(image_path)
        loaded = np.load(path + "/" + video_output + "/" + image_path)
        

        shape = loaded.shape

        while True:
            disp = np.zeros((600, 800, 3), dtype=np.uint8)
            disp[:shape[1], :shape[2], :] = loaded[0]
            cv2.rectangle(disp, (x, y), (x+w, y+h), (255, 255, 255), 1)
            cv2.putText(disp, str(image_path), (35, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('GUI', disp)
            key = cv2.pollKey()
            if key == 32:  # 32 is the ASCII code for spacebar
                print("Spacebar pressed!")
                if shape[1] == 256 or shape[2] == 256:
                    print("already cropped, skipping")
                    cropped_image = loaded
                else:
                    cropped_image = loaded[:, y:y+h, x:x+w, :]
                cropped.append(cropped_image)
                print("Cropped #: ", len(cropped))
                break
        key = cv2.pollKey()
        if key == ord("l"):
            print("Skipping to save")
            break

    print("Press Q to Save and Advance")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            os.mkdir("D:/dataset/" + args.image + "/" + video_output + "/")
            if len(cropped) == 0:
                print("No crop b/c already cropped, skipping")
                break
            for idx, img in enumerate(cropped):
                t_img = torch.from_numpy(img).float()
                t_permuted = t_img.permute(0, 3, 1, 2)  #T, C, H, W
                t_img_resized = resize(t_permuted)
                t_img_save = t_img_resized.permute(1, 0, 2, 3) #C, T, H, W
                if (idx == 0):
                    print(t_img_resized.shape)
                    print(t_img_save.shape)
                np.save("D:/dataset/" + args.image + "/" + video_output + "/" + video_output + "_" + str(idx).zfill(4), t_img_save)
            break

    

# # print file locations
# print("Location for cam1: " + path1)
# print("Location for cam2: " + path2)
# print("Location for cam3: " + path3)
# print("Location for cam4: " + path4)

# # create video capture objects
# vid1 = cv2.VideoCapture(path1)
# vid2 = cv2.VideoCapture(path2)
# vid3 = cv2.VideoCapture(path3)
# vid4 = cv2.VideoCapture(path4)


# #######################################################################
# # capture frames
# #######################################################################


# frames1 = []
# frames2 = []
# frames3 = []
# frames4 = []




# while(vid1.isOpened() and vid2.isOpened() and vid3.isOpened() and vid4.isOpened()):
#     # Capture each frame
#     ret1, frame1 = vid1.read()
#     ret2, frame2 = vid2.read()
#     ret3, frame3 = vid3.read()
#     ret4, frame4 = vid4.read()

#     if ret1: # if frame exists (should for all)
#         frames1.append(frame1)
#         frames2.append(frame2)
#         frames3.append(frame3)
#         frames4.append(frame4)
#     else:
#         break

# print("Read: ", len(frames1), " frames")


# #######################################################################
# # saving initialization
# #######################################################################


# start_time = time.time()

# fps = 250 #ms
# title = name.split(".")[0]

# images1 = []
# images2 = []
# images3 = []
# images4 = []


# #######################################################################
# # save as npy files
# #######################################################################


# counter = 0
# position = 0
# while position < len(frames1)-4:
#     print(str(position) + "\r", end="")

#     loaded = np.load("../dataset/images/" + sequence)

#     # create new gif (multi image)
#     image1 = np.zeros((3, 480, 704, 3), dtype=np.uint8)

#     # add gifs to list for annotation
#     images1.append(loaded)
   

#     # save gif
#     #np.save("../dataset/images1/" + title + "/" + title + "_" + str(counter).zfill(5), image1)
#     #np.save("../dataset/images2/" + title + "/" + title + "_" + str(counter).zfill(5), image2)
#     #np.save("../dataset/images3/" + title + "/" + title + "_" + str(counter).zfill(5), image3)
#     #np.save("../dataset/images4/" + title + "/" + title + "_" + str(counter).zfill(5), image4)


#     # break
#     position += 3
#     counter += 1

#     # Press Q on keyboard to exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break



# #######################################################################################################
# #######################################################################################################
# #
# #           Annotation GUI
# #
# #######################################################################################################
# #######################################################################################################

# # init gui window
# gui = cv2.namedWindow("Window", cv2.WINDOW_KEEPRATIO)

# disp = np.zeros((600, 800, 3), dtype=np.uint8)


# while chunk_id < len(images1):


#     # init basics
#     start_time = time.time()
#     elapsed = 0
#     pos = 0
    
#     l = left[chunk_id]
#     r = right[chunk_id]
    


#     #######################################################################
#     # loop through each gif
#     #######################################################################


#     while True:

#         # work by fps
#         if elapsed >= fps:
#             start_time = time.time()
#             elapsed = 0
#             pos += 1
#             # set position (allows for rotation of images in gif)
#             if pos % 3 == 0 and pos != 0:
#                 pos = 0
#         else:
#             disp = np.zeros((600, 800, 3), dtype=np.uint8)

#             # set 4 views
#             o11r = cv2.resize(chunks[0][chunk_id][pos], (352, 240))
#             o12r = cv2.resize(chunks[1][chunk_id][pos], (352, 240))
#             o13r = cv2.resize(chunks[2][chunk_id][pos], (352, 240))
#             o14r = cv2.resize(chunks[3][chunk_id][pos], (352, 240))

#             disp[:240, :352, :] = o11r
#             disp[240:480, :352, :] = o12r
#             disp[:240, 352:704, :] = o13r
#             disp[240:480, 352:704, :] = o14r

#             # basic display things
#             cv2.putText(disp, str(chunk_id), (350, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             cv2.putText(disp, "U", (35, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(disp, "D", (57, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(disp, "L", (79, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(disp, "R", (101, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             cv2.putText(disp, "L", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(disp, "R", (15, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             if l == 0:
#                 cv2.rectangle(disp, (30, 530), (50, 550), (255, 255, 255), -1)
#             elif l == 1:
#                 cv2.rectangle(disp, (52, 530), (72, 550), (0, 255, 0), -1)
#             elif l == 2:
#                 cv2.rectangle(disp, (74, 530), (94, 550), (255, 0, 0), -1)
#             elif l == 3:
#                 cv2.rectangle(disp, (96, 530), (116, 550), (0, 0, 255), -1)
#             elif l == 4:
#                 cv2.rectangle(disp, (118, 530), (138, 550), (255, 0, 255), -1)

#             if r == 0:
#                 cv2.rectangle(disp, (30, 560), (50, 580), (255, 255, 255), -1)
#             elif r == 1:
#                 cv2.rectangle(disp, (52, 560), (72, 580), (0, 255, 0), -1)
#             elif r == 2:
#                 cv2.rectangle(disp, (74, 560), (94, 580), (255, 0, 0), -1)
#             elif r == 3:
#                 cv2.rectangle(disp, (96, 560), (116, 580), (0, 0, 255), -1)
#             elif r == 4:
#                 cv2.rectangle(disp, (118, 560), (138, 580), (255, 0, 255), -1)


#             cv2.imshow('Window', disp)

#         end_time = time.time()

#         elapsed = (end_time - start_time) * 1000


#         #######################################################################
#         # set feet positions
#         #######################################################################

#         polled = cv2.pollKey()

#         if polled == ord('q'):
#             l = 0
#         if polled == ord('w'):
#             l = 1
#         if polled == ord('e'):
#             l = 2
#         if polled == ord('r'):      
#             l = 3
#         if polled == ord('t'):
#             l = 4
#         if polled == ord('a'):
#             r = 0
#         if polled == ord('s'):
#             r = 1
#         if polled == ord('d'):
#             r = 2
#         if polled == ord('f'):      
#             r = 3
#         if polled == ord('g'):
#             r = 4


#         #######################################################################
#         # forward and backward buttons
#         #######################################################################


#         if polled == 32:
#             if chunk_id < len(images1) - 1:
#                 left[chunk_id] = l
#                 right[chunk_id] = r
#                 print("continue")
#                 break
#             else:
#                 print("At last gif. Can't go forward")
#                 break

#         if polled == ord('b'):
#             if chunk_id > 0:

#                 chunk_id -= 2
#                 print("back")
#                 break
#             else:
#                 print("At first gif. Can't go back")

#     chunk_id += 1

# # convert left and right to csv files and save
# left_label_path = "../dataset/labels/" + title + "_left.csv"
# right_label_path = "../dataset/labels/" + title + "_right.csv"

# np.savetxt(left_label_path, left, delimiter=",")
# np.savetxt(right_label_path, right, delimiter=",")

# os.mkdir("../dataset/images1/" + title)
# os.mkdir("../dataset/images2/" + title)
# os.mkdir("../dataset/images3/" + title)
# os.mkdir("../dataset/images4/" + title)

# for i in range(len(images1)):
#     # save gif
#     np.save("../dataset/images1/" + title + "/" + title + "_" + str(i).zfill(5), images1[i])
#     np.save("../dataset/images2/" + title + "/" + title + "_" + str(i).zfill(5), images2[i])
#     np.save("../dataset/images3/" + title + "/" + title + "_" + str(i).zfill(5), images3[i])
#     np.save("../dataset/images4/" + title + "/" + title + "_" + str(i).zfill(5), images4[i])
                


# print("Completed. Data saved")
