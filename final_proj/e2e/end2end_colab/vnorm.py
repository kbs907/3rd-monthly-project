#!/usr/bin/env python

import cv2, glob, os
import numpy as np

video_files = glob.glob("*.mkv")
caps = [ cv2.VideoCapture(name) for name in video_files ]

cap_length = max([ int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps ])

if not os.path.isdir("./image/") :
    os.mkdir("./image/")

for i in range(cap_length) :
    for cap in caps :
        cap_name = str(cap).split(" ")[1][:-1]

        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.resize(img, dsize=(200, 150))
        img = img[84:,:]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
	edge_img = cv2.Canny(np.uint8(blur_gray), 50, 150)
        cv2.imwrite("image/"+str(i)+"-"+cap_name+".jpg", edge_img)

for cap, video_file in zip(caps, video_files) :
    current_name = str(video_file).split(".")[0]
    print(current_name)
    print(current_name.split("_")[0])
    cap_name = str(cap).split(" ")[1][:-1]
    print(cap_name)
    os.rename(current_name+".mkv", cap_name+".mkv")
    print(current_name+".csv")
    os.rename(current_name+".csv", cap_name+".csv")

for cap in caps :
    cap.release()

cv2.destroyAllWindows()
