import cv2
import os
import numpy as np
import argparse
import time


gui = cv2.namedWindow("Window", cv2.WINDOW_KEEPRATIO)

o11 = np.load("../dataset/images/output1_100000000.npy")
o12 = np.load("../dataset/images/output1_100000001.npy")
o13 = np.load("../dataset/images/output1_100000002.npy")
o14 = np.load("../dataset/images/output1_100000003.npy")
o21 = np.load("../dataset/images/output1_100000004.npy")
o22 = np.load("../dataset/images/output1_100000005.npy")
o23 = np.load("../dataset/images/output1_100000006.npy")
o24 = np.load("../dataset/images/output1_100000007.npy")

disp = np.zeros((600, 800, 3), dtype=np.uint8)

fps = 250 #ms

chunk = [(o11, o12, o13, o14), (o21, o22, o23, o24)]
names = ["0001", "0002"]
    
for chunk_id in range(len(chunk)):
    # for i in range(210):
        # print(i)

    
    start_time = time.time()
    elapsed = 0
    pos = 0
    l = 4
    r = 4


    while True:

        if elapsed >= fps:
            start_time = time.time()
            elapsed = 0
            pos += 1
            if pos % 3 == 0 and pos != 0:
                pos = 0
        else:
            disp = np.zeros((600, 800, 3), dtype=np.uint8)

            o11r = cv2.resize(chunk[chunk_id][0][pos], (352, 240))
            o12r = cv2.resize(chunk[chunk_id][1][pos], (352, 240))
            o13r = cv2.resize(chunk[chunk_id][2][pos], (352, 240))
            o14r = cv2.resize(chunk[chunk_id][3][pos], (352, 240))

            disp[:240, :352, :] = o11r
            disp[240:480, :352, :] = o12r
            disp[:240, 352:704, :] = o13r
            disp[240:480, 352:704, :] = o14r

            cv2.putText(disp, names[chunk_id], (350, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.putText(disp, "U", (35, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(disp, "D", (57, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(disp, "L", (79, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(disp, "R", (101, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.putText(disp, "L", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(disp, "R", (15, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if l == 0:
                cv2.rectangle(disp, (30, 530), (50, 550), (255, 255, 255), -1)
            elif l == 1:
                cv2.rectangle(disp, (52, 530), (72, 550), (0, 255, 0), -1)
            elif l == 2:
                cv2.rectangle(disp, (74, 530), (94, 550), (255, 0, 0), -1)
            elif l == 3:
                cv2.rectangle(disp, (96, 530), (116, 550), (0, 0, 255), -1)
            elif l == 4:
                cv2.rectangle(disp, (118, 530), (138, 550), (255, 0, 255), -1)

            if r == 0:
                cv2.rectangle(disp, (30, 560), (50, 580), (255, 255, 255), -1)
            elif r == 1:
                cv2.rectangle(disp, (52, 560), (72, 580), (0, 255, 0), -1)
            elif r == 2:
                cv2.rectangle(disp, (74, 560), (94, 580), (255, 0, 0), -1)
            elif r == 3:
                cv2.rectangle(disp, (96, 560), (116, 580), (0, 0, 255), -1)
            elif r == 4:
                cv2.rectangle(disp, (118, 560), (138, 580), (255, 0, 255), -1)


            cv2.imshow('Window', disp)

        end_time = time.time()

        elapsed = (end_time - start_time) * 1000
        # print(pos)
        # time.sleep((fps - elapsed)/1000)

        # Press Q on keyboard to exit

        polled = cv2.pollKey()

        if polled == ord('q'):
            l = 0
        if polled == ord('w'):
            l = 1
        if polled == ord('e'):
            l = 2
        if polled == ord('r'):      
            l = 3
        if polled == ord('t'):
            l = 4
        if polled == ord('a'):
            r = 0
        if polled == ord('s'):
            r = 1
        if polled == ord('d'):
            r = 2
        if polled == ord('f'):      
            r = 3
        if polled == ord('g'):
            r = 4

        if polled == 32:
            print("continue")
            break

        if polled == 8:
            print("back")

        
