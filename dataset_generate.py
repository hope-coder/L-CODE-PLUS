#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 15:30
# @Author  : ZWP
# @Desc    : 
# @File    : dataset_generate.py

import random
import pandas as pd
import shap
from river import synth
import numpy as np
import cv2
import os


def getDateset(dataset="SEA", num=1000, drift=None):
    if drift is None:
        drift = [900]
    if dataset == "SEA":
        normal_size = int(drift[0])
        train_data = normal_data(normal_size)
        drift = drift_data(num - normal_size)
        data = pd.concat([train_data, drift], axis=0, ignore_index=True)
        X = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]
        return X, y
    elif dataset == "XGBoost":
        X, y = shap.datasets.boston()
        return X, y
    elif dataset == 'electricity':
        data = pd.read_csv('./dataset/electricity-normalized.csv')
        data.drop('date', axis=1, inplace=True)
        data = data.iloc[0:43200, ]
        X = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]
        return X, y
    elif dataset == 'dog_cat':
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
        normal_size = int(drift[0])
        for i in range(normal_size, len(data.index)):
            temp = cv2.GaussianBlur(data.iloc[i, 0][0], (9, 9), 1.5)
            data.iloc[i, 0] = [temp]
        X = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]
        return X, y
    elif dataset == 'phishing':
        df = pd.read_csv('./dataset/phishing.csv')
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        data_X = df.iloc[:, 1:-1]
        columns = list(X.columns)
        for i in range(X.shape[1]):
            if len(set(X.iloc[:, i])) == 2:
                continue
            else:
                dummies = pd.get_dummies(X[columns[i]], prefix=columns[i])
                data_X.drop(columns=columns[i], axis=1, inplace=True)
                data_X = data_X.join(dummies)
        data_X[data_X < 0] = 0
        y[y < 0] = 0
        return data_X.iloc[0:11000, ], y.iloc[0:11000, ]

    elif dataset == "tree":
        data = synth.ConceptDriftStream(stream=synth.RandomTree(seed_tree=42, seed_sample=42, n_classes=2,
                                                                n_num_features=2, n_cat_features=2,
                                                                n_categories_per_feature=2, max_tree_depth=6,
                                                                first_leaf_level=3, fraction_leaves_per_level=0.15),
                                        drift_stream=synth.RandomTree(seed_tree=65, seed_sample=48, n_classes=2,
                                                                      n_num_features=2, n_cat_features=2,
                                                                      n_categories_per_feature=2, max_tree_depth=6,
                                                                      first_leaf_level=3,
                                                                      fraction_leaves_per_level=0.15),
                                        position=drift[0])
        X_list = []
        y_list = []
        for x, y in data.take(num):
            X_list.append(x)
            y_list.append(y)
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        return X, y

    else:
        data = synth.ConceptDriftStream(
            stream=synth.RandomRBF(n_classes=2, n_features=4, n_centroids=10, seed_sample=42, seed_model=42),
            drift_stream=synth.RandomRBF(n_classes=2, n_features=4, n_centroids=10, seed_sample=92, seed_model=92,
                                              ), position=drift[0])
        X_list = []
        y_list = []
        for x, y in data.take(num):
            X_list.append(x)
            y_list.append(y)
        X = pd.DataFrame(X_list)
        y = np.array(y_list)
        return X, y


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


def gen_y(X):
    y = []
    for i in range(X.shape[0]):
        x = X.iloc[i,]
        if x[0] + x[1] > 7:
            y.append(False)
        else:
            y.append(True)
    return y


def normal_data(num):
    f1_1 = [random.uniform(0, 5) for _ in range(int(num * 0.1))]
    f1_2 = [random.uniform(5, 10) for _ in range(int(num * 0.9))]
    f2 = [random.uniform(0, 10) for _ in range(num)]
    f3 = [random.uniform(0, 10) for _ in range(num)]
    f1 = f1_2 + f1_1
    random.shuffle(f1)
    tmp = list(zip(f1, f2, f3))
    x = [list(i) for i in tmp]
    df = pd.DataFrame(x)
    y = gen_y(df)
    df['y'] = y
    return df


def drift_data(num):
    drift_f1_1 = [random.uniform(0, 5) for _ in range(int(num * 0.9))]
    drift_f1_2 = [random.uniform(5, 10) for _ in range(int(num * 0.1))]
    drift_f2 = [random.uniform(0, 10) for _ in range(num)]
    drift_f3_1 = [random.uniform(0, 5) for _ in range(int(num * 0.1))]
    drift_f3_2 = [random.uniform(5, 10) for _ in range(int(num * 0.9))]
    drift_f1 = drift_f1_2 + drift_f1_1
    random.shuffle(drift_f1)
    drift_f3 = drift_f3_2 + drift_f3_1
    random.shuffle(drift_f3)
    drift_temp = list(zip(drift_f1, drift_f2, drift_f3))
    drift_x = [list(i) for i in drift_temp]
    drift_Data = pd.DataFrame(drift_x)
    drift_y = gen_y(drift_Data)
    drift_Data['y'] = drift_y
    return drift_Data
