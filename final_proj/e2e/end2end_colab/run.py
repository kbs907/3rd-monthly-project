#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from model import end2end

import glob, csv, random, time, io, dill, os, cv2

import numpy as np
from PIL import Image

def study_model_load(epoch, batch_cnt, model, device):
    LoadPath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+".pth"
    with open(LoadPath_main, 'rb') as f :
        LoadBuffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(LoadBuffer, map_location=device))
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = end2end().to(device)
#main_model_000012_000073.pth
net = study_model_load(11, 73, net, device)

input_file = "0x7f438d396690.mkv"
cap = cv2.VideoCapture(input_file)
src = cv2.imread("wheel.png", cv2.IMREAD_COLOR)
h, w, ch = src.shape
 
while cap.isOpened():
    retval, oframe = cap.read()
    if not(retval):
        break
    
    frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(img, dsize=(200, 112))
    frame = frame[46:,:]
    frame = frame.transpose((2,0,1)) / 255.0
    t_frame = torch.FloatTensor([frame]).to(device)

    angle = net(t_frame)

    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    dst = cv2.warpAffine(src, matrix, (w, h))

    cv2.imshow('original', oframe)
    cv2.imshow('wheel', dst)
    cv2.waitKey(1)

cap.release()	  
cv2.destroyAllWindows()
