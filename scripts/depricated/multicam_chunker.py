# Run from within scripts folder

import cv2
import os
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser(
                    prog='Video Annotato',
                    description='Chunks video and annotates',
                    epilog='Help yourself lol')

parser.add_argument('vid_name')           # positional argument
args = parser.parse_args()


#change here!
path1 = "../dataset/cam1/" + args.vid_name
path2 = "../dataset/cam2/" + args.vid_name
path3 = "../dataset/cam3/" + args.vid_name
path4 = "../dataset/cam4/" + args.vid_name

name = args.vid_name

# print file locations
print("Location for cam1: " + path1)
print("Location for cam2: " + path2)
print("Location for cam3: " + path3)
print("Location for cam4: " + path4)

# create video capture objects
vid1 = cv2.VideoCapture(path1)
vid2 = cv2.VideoCapture(path2)
vid3 = cv2.VideoCapture(path3)
vid4 = cv2.VideoCapture(path4)



frames1 = []
frames2 = []
frames3 = []
frames4 = []




while(vid1.isOpened() and vid2.isOpened() and vid3.isOpened() and vid4.isOpened()):
    # Capture each frame
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()
    ret3, frame3 = vid3.read()
    ret4, frame4 = vid4.read()

    if ret1: # if frame exists (should for all)
        frames1.append(frame1)
        frames2.append(frame2)
        frames3.append(frame3)
        frames4.append(frame4)
    else:
        break

print("Read: ", len(frames1), " frames")

start_time = time.time()

fps = 250 #ms
title = name.split(".")[0]

os.mkdir("../dataset/images1/" + title)
os.mkdir("../dataset/images2/" + title)
os.mkdir("../dataset/images3/" + title)
os.mkdir("../dataset/images4/" + title)


counter = 0
position = 0
while position < len(frames1)-4:
    print(str(position) + "\r", end="")
    # create new gif (multi image)
    image1 = np.zeros((3, 480, 704, 3), dtype=np.uint8)
    image2 = np.zeros((3, 480, 704, 3), dtype=np.uint8)
    image3 = np.zeros((3, 480, 704, 3), dtype=np.uint8)
    image4 = np.zeros((3, 480, 704, 3), dtype=np.uint8)

    # add frames to gif
    for i in range(3):
        image1[i] = frames1[position + i]
        image2[i] = frames2[position + i]
        image3[i] = frames3[position + i]
        image4[i] = frames4[position + i]


    # save gif
    np.save("../dataset/images1/" + title + "/" + title + "_" + str(counter).zfill(5), image1)
    np.save("../dataset/images2/" + title + "/" + title + "_" + str(counter).zfill(5), image2)
    np.save("../dataset/images3/" + title + "/" + title + "_" + str(counter).zfill(5), image3)
    np.save("../dataset/images4/" + title + "/" + title + "_" + str(counter).zfill(5), image4)


    # break
    position += 3
    counter += 1

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print("Finished saving images")