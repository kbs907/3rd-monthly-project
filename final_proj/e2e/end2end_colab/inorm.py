#!/usr/bin/env python

import cv2
 
video_file = 'drive_1_e48.mkv'
 
cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, img = cap.read()
    print(img.shape)
    if not ret:
        break
    img = cv2.resize(img, dsize=(200, 150)) # 1280,720 -> 200,112 | 640,480 -> 200, 150
    img = img[84:,:] / 255.0	# 112 - 66 = 46 , 150 - 66 = 84
    cv2.imshow(video_file, img)
    cv2.waitKey(1)
 
cap.release()
cv2.destroyAllWindows()
