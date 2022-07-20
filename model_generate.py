# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2022/6/24 11:29
# # @Author  : ZWP
# # @Desc    :
# # @File    : model_generate.py
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.python.keras.layers.embeddings import Embedding
# from tqdm import tqdm
# import shap
# import tensorflow.compat.v1.keras.backend as K
# import tensorflow as tf
# import numpy as np
#
# tf.compat.v1.disable_eager_execution()
#
# # print the JS visualization code to the notebook
# import dataset_generate
#
# shap.initjs()
#
#
# def test():
#     X, y = shap.datasets.adult()
#     # 进行数据集的归一化处理
#     dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
#     for k, dtype in dtypes:
#         if dtype == "float32":
#             X[k] -= X[k].mean()
#             X[k] /= X[k].std()
#     # 划分数据集
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
#     input_els = []
#     encoded_els = []
#     for k, dtype in dtypes:
#         input_els.append(Input(shape=(1,)))
#         # 从刚刚加入的Input上创建一个嵌入层加平铺层，否则不加入嵌入层
#         if dtype == "int8":
#             e = Flatten()(Embedding(X_train[k].max() + 1, 1)(input_els[-1]))
#         else:
#             e = input_els[-1]
#         encoded_els.append(e)
#     encoded_els = concatenate(encoded_els)
#     layer1 = Dropout(0.5)(Dense(100, activation="relu")(encoded_els))
#     layer2 = Dense(10)(layer1)
#     out = Dense(1)(layer2)
#
#     # train model
#     regression = Model(inputs=input_els, outputs=[out])
#     regression.summary()
#     regression.compile(optimizer="adam", loss='binary_crossentropy')
#     regression.fit(
#         [X_train[k].values for k, t in dtypes],
#         y_train,
#         epochs=50,
#         batch_size=512,
#         shuffle=True,
#         validation_data=([X_valid[k].values for k, t in dtypes], y_valid)
#     )
#
#     def f(X):
#         print(X)
#         return regression.predict([X[:, i] for i in range(X.shape[1])]).flatten()
#
#     explainer = shap.KernelExplainer(f, X.iloc[:50, :])
#     shap_values = explainer.shap_values(X.iloc[299, :], nsamples=500)
#     shap.force_plot(explainer.expected_value, shap_values, X.iloc[299, :])
#
#
# def getExplainer(model, X, y):
#     # 进行数据集的归一化处理
#     dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
#     for k, dtype in dtypes:
#         if dtype == "float32":
#             X[k] -= X[k].mean()
#             X[k] /= X[k].std()
#     # 划分数据集
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
#     input_els = []
#     encoded_els = []
#     for k, dtype in dtypes:
#         input_els.append(Input(shape=(1,)))
#         # 从刚刚加入的Input上创建一个嵌入层加平铺层，否则不加入嵌入层
#         if dtype == "int8":
#             e = Flatten()(Embedding(X_train[k].max() + 1, 1)(input_els[-1]))
#         else:
#             e = input_els[-1]
#         encoded_els.append(e)
#     encoded_els = concatenate(encoded_els)
#     layer1 = Dropout(0.5)(Dense(100, activation="relu")(encoded_els))
#     layer2 = Dense(10, name="drift")(layer1)
#     out = Dense(1)(layer2)
#
#     # train model
#     regression = Model(inputs=input_els, outputs=[out])
#     regression.summary()
#     regression.compile(optimizer="adam", loss='binary_crossentropy',
#                        metrics=['accuracy'])
#     regression.fit(
#         [X_train[k].values for k, t in dtypes],
#         y_train,
#         epochs=20,
#         batch_size=512,
#         shuffle=True,
#         validation_data=([X_valid[k].values for k, t in dtypes], y_valid)
#     )
#
#     def map2layer(x, layer_name):
#         x_copy = x.copy()
#         feed_dict = dict(zip(regression.inputs, [np.reshape(x_copy[k].values, (-1, 1)) for k, t in dtypes]))
#         return K.get_session().run(regression.get_layer(layer_name).input, feed_dict)
#
#     def f(X):
#         print(X)
#         return regression.predict([X[:, i] for i in range(X.shape[1])]).flatten()
#
#     explainer = shap.GradientExplainer((regression.get_layer("drift").input, regression.layers[-1].output),
#                                        map2layer(X.copy(), "drift"))
#     shap_values = explainer.shap_values(map2layer(X.iloc[[299], :], "drift"), nsamples=500)
#     shap.force_plot(shap_values, X.iloc[299, :])
#
#
# if __name__ == '__main__':
#     X, y = dataset_generate.getDateset("SEA", 10000, [9000])
#     getExplainer("CNN", X, y)
