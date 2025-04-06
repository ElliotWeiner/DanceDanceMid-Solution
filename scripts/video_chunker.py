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

path = args.vid_name
name, _ = os.path.splitext(os.path.basename(path))
print(path, name)
# print("Files and directories in '", path, "' :")
# prints all files

print("Location: vids/" + path)

# for vid_name in dir_list:
vid = cv2.VideoCapture(path)

frames = []

# while(vid.isOpened()):
# # Capture each frame
#     ret, frame = vid.read()
#     cv2.imshow('Frame', frame)
    
#     # Press Q on keyboard to exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

while(vid.isOpened()):
    # Capture each frame
    ret, frame = vid.read()
    if ret:
        frames.append(frame)
    else:
        break

print("Read: ", len(frames), " frames")

start_time = time.time()

fps = 250 #ms

counter = 0
position = 0
while position < len(frames)-4:
    print(position)
    image = np.zeros((3, 480, 704, 3), dtype=np.uint8)
    for i in range(3):
        # print(i)
        # start_time = time.time()
        # cv2.imshow('Frame', frames[i+position])
        # end_time = time.time()

        # elapsed = (end_time - start_time) * 1000
        # print(i, fps - elapsed)
        # time.sleep((fps - elapsed)/1000)
        # print(frames[i+position].shape)
        image[i] = frames[position + i]


    np.save("../dataset/images/" + name + str(counter).zfill(8), image)

    # loaded = np.load("dataset/images/" + name + str(counter) + ".npy")

    # print(image == loaded)

    # break
    position += 3
    counter += 1

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break