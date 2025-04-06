import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
                    prog='Video Annotato',
                    description='Chunks video and annotates',
                    epilog='Help yourself lol')

parser.add_argument('datadir')           # positional argument
args = parser.parse_args()

path = args.datadir
dir_list = os.listdir(path)
# print("Files and directories in '", path, "' :")
print(dir_list)
# prints all files


for sequence in dir_list:

    # frames = []
    loaded = np.load("../dataset/images/" + sequence)
    print(sequence)
    print(loaded.dtype)
    # while(vid.isOpened()):
    # # Capture each frame
    #     ret, frame = vid.read()
    #     cv2.imshow('Frame', frame)
        
    #     # Press Q on keyboard to exit
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
        
    for pos in range(3):
        # for i in range(210):
            # print(i)
        print(pos)
        cv2.imshow('Frame', loaded[pos])

        # Press Q on keyboard to exit

        if cv2.waitKey(0):
            continue    

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break