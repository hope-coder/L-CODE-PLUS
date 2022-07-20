#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 11:26
# @Author  : ZWP
# @Desc    : 
# @File    : test.py

import random
import pandas as pd
import shap
from river import synth
import numpy as np
import cv2
import os


def get_image_data(image_path):
    catfiles = os.listdir(image_path)
    count = 0
    for root, dirs, files in os.walk(image_path):  # 遍历统计
        for each in files:
            count += 1
    image_data = np.empty((count, 150, 150, 3))
    i = 0
    for file in catfiles:
        img = cv2.imread(image_path + "/" + file)
        low_image = cv2.resize(img, (150, 150))
        image_data[i] = low_image
        i = i + 1
    return image_data


def make_image_drift(data, drifts, drift_type):
    drifts.append(data.size)

    for index, drift in enumerate(drifts):
        # 不是最后一个元素时
        if index < drifts[-1]:
            for i in range(drift, drifts[index + 1]):
                if drift_type[index]== "gauss":
                    temp = cv2.GaussianBlur(data.iloc[i, 0][0], (9, 9), 1.5)
                    data.iloc[i, 0] = [temp]
                elif drift_type[index] == "brightness":
                    temp = liner_trans(data.iloc[i, 0][0], 1.2)
                elif drift_type[index] == "darken":
                    temp = liner_trans(data.iloc[i, 0][0], 0.8)

    return data

def liner_trans(img,gamma):#gamma大于1时图片变亮，小于1图片变暗
    img=np.float32(img)*gamma//1
    img[img>255]=255
    img=np.uint8(img)
    return img

if __name__ == '__main__':
    cat_path = './Dog_Cat/Cat'
    dog_path = './Dog_Cat/Dog'
    cat_data = get_image_data(cat_path)
    dog_data = get_image_data(dog_path)
    data = pd.DataFrame(None, columns=['image', 'label'])
    for cat in cat_data:
        data.loc[len(data.index)] = [[cat], 'cat']
    for dog in dog_data:
        data.loc[len(data.index)] = [[dog], 'dog']
    random.shuffle(data)
