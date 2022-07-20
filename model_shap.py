#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:06
# @Author  : ZWP
# @Desc    :
# @File    : model_shap.py
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.embeddings import Embedding
from tqdm import tqdm
import shap
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

import dataset_generate

tf.compat.v1.disable_eager_execution()


class object_model:
    def __init__(self, windows_size, test=0.2, dataset="SEA", classification=True, shap_class=0, external=False):

        shap.initjs()  # notebook环境下，加载用于可视化的JS代码
        self.classification = classification
        # 我们先训练好一个XGBoost model
        self.windows_size = windows_size
        self.shap_class = shap_class

        # 设置漂移点
        self.drift = [8000]

        # 数据准备
        X, y = dataset_generate.getDateset(dataset, 10000, self.drift)
        data_size = X.shape[0]
        self.train_size = int(data_size * (1 - test))
        self.X_train = X[:self.train_size]
        self.X_test = X[self.train_size:]
        self.y_train = y[:self.train_size]
        self.y_test = y[self.train_size:]

        self.windows_number = 0
        self.windows_max_number = int((data_size - self.train_size) / self.windows_size)

        test_size = data_size - self.train_size
        if windows_size * 2 >= test_size:
            raise Exception("窗口太大")

        # 判断是否使用外置检测器
        self.external = external
        if self.external:
            self.ex_detector = external_detector("drift")
            self.ex_detector.build_detector(self.X_train, self.y_train)
            self.X_middle_train = self.ex_detector.layer2df(self.X_train)

        if dataset == "SEA":

            log_reg = RandomForestClassifier(n_estimators=20)
            log_reg.fit(self.X_train, self.y_train)
            self.model = log_reg
            self.explainer = shap.KernelExplainer(log_reg.predict_proba, self.X_train)
            self.explainer = shap.TreeExplainer(log_reg)
            print("测试效果" + str(log_reg.score(self.X_train, self.y_train)))
        elif dataset == "XGBoost":

            self.model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(self.X_train, label=self.y_train), 100)
            self.explainer = shap.TreeExplainer(self.model)
            # print("测试效果" + str(xgboost.score))
        else:

            log_reg = RandomForestClassifier(n_estimators=20)
            log_reg.fit(self.X_train, self.y_train)
            self.model = log_reg
            print("测试效果" + str(log_reg.score(self.X_train, self.y_train)))
            # self.explainer = shap.KernelExplainer(log_reg.predict_proba, self.X_train)
            self.explainer = shap.TreeExplainer(log_reg)

    def getNextWindows(self):
        is_drift = False
        if self.windows_number < self.windows_max_number:
            for drift_point in self.drift:
                if (self.windows_size * self.windows_number) <= drift_point - self.train_size < self.windows_size * (
                        self.windows_number + 1):
                    is_drift = True
            X_detect = self.X_test[
                       (self.windows_size * self.windows_number): self.windows_size * (self.windows_number + 1)]
            y_detect = self.y_test[
                       (self.windows_size * self.windows_number): self.windows_size * (self.windows_number + 1)]
            all_acc = [self.model.score(X_detect, y_detect)]
            print("\n 窗口号：", self.windows_number, "数据段：", self.train_size + self.windows_size * self.windows_number,
                  self.train_size + self.windows_size * (self.windows_number + 1), "当前窗口结果：", all_acc[0])

            if self.external:

                # 对外置检测器的准确度做保证
                all_acc.append(self.ex_detector.get_score(X_detect, y_detect)[1])
                print("外置检测器置信度：", self.ex_detector.get_score(X_detect, y_detect))
                shap_values = self.ex_detector.get_shap_values(X_detect)
                X_detect = self.ex_detector.layer2df(X_detect)

            else:
                shap_values = self.explainer.shap_values(X_detect)
            # 下一窗口应该取的数据段
            self.windows_number = self.windows_number + 1
            if self.classification:
                return False, X_detect, shap_values[self.shap_class], is_drift, all_acc
            else:
                return False, X_detect, shap_values, is_drift, all_acc

        else:
            return True, None, None, None, None

    def getTrain(self):
        if self.external:
            shap_values = self.ex_detector.get_shap_values(self.X_train)
        else:
            shap_values = self.explainer.shap_values(self.X_train)
        if self.classification:
            return self.X_train, shap_values[self.shap_class]
        else:
            return self.X_train, shap_values

    def reTrain(self, windows_number):
        X_retrain = self.X_test[
                    (self.windows_size * windows_number): self.windows_size * (windows_number + 1)]
        y_retrain = self.y_test[
                    (self.windows_size * windows_number): self.windows_size * (windows_number + 1)]
        self.model.fit(X_retrain, y_retrain)
        if not self.external:
            self.explainer = shap.TreeExplainer(model=self.model)
            return X_retrain, self.explainer.shap_values(X_retrain)[self.shap_class]
        else:
            # 如果重训练则重新构造检测器
            self.ex_detector.build_detector(X_retrain, y_retrain)
            self.X_middle_train = self.ex_detector.layer2df(X_retrain)
            return self.X_middle_train, self.ex_detector.get_shap_values(X_retrain)[self.shap_class]


class external_detector:
    def __init__(self, layer_name):
        self.dtypes = None
        self.model = None
        self.X_middle_train = None
        self.layer_name = layer_name

    def build_detector(self, X, y, model_type="normal"):
        # 进行数据集的归一化处理
        self.dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
        for k, dtype in self.dtypes:
            if dtype == "float32":
                X[k] -= X[k].mean()
                X[k] /= X[k].std()
        # 划分数据集
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
        input_els = []
        encoded_els = []
        for k, dtype in self.dtypes:
            input_els.append(Input(shape=(1,)))
            # 从刚刚加入的Input上创建一个嵌入层加平铺层，否则不加入嵌入层
            if dtype == "int8":
                e = Flatten()(Embedding(X_train[k].max() + 1, 1)(input_els[-1]))
            else:
                e = input_els[-1]
            encoded_els.append(e)
        encoded_els = concatenate(encoded_els)
        layer1 = Dropout(0.5)(Dense(10, activation="relu", name="input")(encoded_els))
        layer2 = Dense(30, name="drift", activation="relu")(layer1)
        layer3 = Dense(5)(layer2)
        out = Dense(1, name="output", activation="sigmoid")(layer3)

        # train model
        self.model = Model(inputs=input_els, outputs=[out])
        self.model.summary()
        self.model.compile(optimizer="adam", loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(
            [X_train[k].values for k, t in self.dtypes],
            y_train,
            epochs=100,
            batch_size=512,
            shuffle=True,
            validation_data=([X_valid[k].values for k, t in self.dtypes], y_valid)
        )
        self.X_middle_train = self.map2layer(X.copy(), self.layer_name)

        self.explainer = shap.GradientExplainer(
            (self.model.get_layer(self.layer_name).input, self.model.layers[-1].output),
            self.map2layer(X.copy(), self.layer_name))

        shap_values = self.explainer.shap_values(self.map2layer(X, self.layer_name), nsamples=500)
        # shap.force_plot(shap_values, X.iloc[299, :])

        return shap_values

    def map2layer(self, x, layer_name):
        x_copy = x.copy()
        self.preprocessing(x_copy)

        feed_dict = dict(zip(self.model.inputs, [np.reshape(x_copy[k].values, (-1, 1)) for k, t in self.dtypes]))
        return K.get_session().run(self.model.get_layer(layer_name).input, feed_dict)

    def layer2df(self, x):
        layer = self.map2layer(x, self.layer_name)
        layer = pd.DataFrame(layer)
        return layer

    def get_shap_values(self, X):
        return self.explainer.shap_values(self.map2layer(X, self.layer_name), nsamples=500)

    def get_score(self, X, y):
        x = self.preprocessing(X)
        return self.model.evaluate([x[k].values for k, t in self.dtypes], y)

    def preprocessing(self, x_copy):
        # 同样归一化
        for k, dtype in self.dtypes:
            if dtype == "float32":
                x_copy[k] -= x_copy[k].mean()
                x_copy[k] /= x_copy[k].std()
        return x_copy
