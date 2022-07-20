#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 16:37
# @Author  : ZWP
# @Desc    : 
# @File    : draw.py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid", {"font.sans-serif": ['KaiTi', 'Arial']})


class drift_visualization():
    def __init__(self, feature_select, alpha):
        self.alpha = alpha
        self.drift = []  # 记录实际漂移的位置
        self.warning = {}  # 记录检测出漂移的位置
        self.all_warning = []
        self.accuracy = []  # 记录运行过程中准确率的变化
        self.ex_accuracy = []
        self.p_value = {}  # 记录判别标准p值的变化趋势
        self.windows_number = []

        for feature in feature_select:
            self.p_value[feature] = []
            self.warning[feature] = []

    def add_acc(self, windows_number, acc):
        self.accuracy.append(acc)
        self.windows_number.append(windows_number)

    def add_ex_acc(self, windows_number, acc):
        self.ex_accuracy.append(acc)

    def add_p_value(self, windows_number, feature_name, p_value, is_drift):
        self.p_value[feature_name].append(p_value)
        if is_drift:
            self.warning[feature_name].append(windows_number)
            self.all_warning.append(windows_number)

    def add_drift(self, windows_number):
        self.drift.append(windows_number)

    def getScore(self, delay=2):
        temDrift = self.drift.copy()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        size = len(self.windows_number)
        # 对每一个被检测出的窗口做循环
        for window in self.all_warning:
            # 开始默认阳值为假
            positive = False
            for trueDrift in temDrift:
                # 若在该检测结果之前存在真实漂移则检测成功
                if trueDrift <= window <= trueDrift + delay:
                    positive = True
                    # 该值被很好的检测出来了，就去掉它了可以
                    temDrift.remove(trueDrift)
            if positive:
                TP += 1
            else:
                FP += 1
        # 每一个未检测到的漂移都为FN（暂时）
        FN = len(temDrift)
        TN = size - TN - TP - FN
        print("本次测试的结果：", TP, TN, FP, FN)
        return TP, TN, FP, FN

    def do_draw(self, feature):
        mean_acc = np.mean(self.accuracy)
        print("\n 平均准确率：", mean_acc)
        plt.subplot(2, 1, 1)
        plt.xlabel('index')
        plt.ylabel('accuracy/p-value')

        x_alpha = [self.windows_number[0], self.windows_number[-1]]

        y_alpha = [self.alpha, self.alpha]

        plt.vlines(x=self.drift, ymin=0.5, ymax=1, colors='r', linestyles='-',
                   label='drift')
        plt.vlines(x=self.all_warning, ymin=0, ymax=0.5, colors='g', linestyles=':',
                   label='drift_detect')

        plt.plot(x_alpha, y_alpha, lw=0.5, color="g", label="alpha判别")
        plt.plot(self.windows_number, self.accuracy, lw=2, label='accuracy')
        for key in self.p_value.keys():
            plt.plot(self.windows_number, self.p_value[key], lw=1, color="g")

        plt.legend()
        plt.title("漂移检测图")
        plt.subplot(2, 1, 2)
        plt.ylim(0.7, 1)
        plt.plot(self.windows_number, self.accuracy, lw=2, label='accuracy')
        if len(self.ex_accuracy) > 0:
            plt.plot(self.windows_number, self.ex_accuracy, lw=2, label='ex_accuracy', color="g")
        plt.show()
        # fig.savefig('./')
