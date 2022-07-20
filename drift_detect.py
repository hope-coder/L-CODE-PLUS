#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:40
# @Author  : ZWP
# @Desc    : 
# @File    : drift_detect.py
import math
import scipy.stats as stats
import pandas as pd
import numpy as np


class drift_detect:
    # 在漂移检测之间首先要获取训练集的上下界，漂移检测只在上下界的范围内进行
    # 超出上下界后是异常值检测的工作了。
    def __init__(self, X_train, window_size, threshold=0.99, alpha=0.01, external=False):
        self.external=external
        self.alpha = alpha
        self.threshold = threshold
        self.window_size = window_size
        self.bin_num = int(window_size ** 0.5)
        self.columns = X_train.columns
        self.feature_max_values = {}
        self.feature_min_values = {}
        self.feature_interval = {}
        for feature in self.columns:
            feature_values = X_train[feature]
            self.feature_max_values[feature] = max(feature_values)
            self.feature_min_values[feature] = min(feature_values)
            self.feature_interval[feature] = (self.feature_max_values[feature] - self.feature_min_values[
                feature]) / self.bin_num

    # 生成特征空间与SHAP值的分布数据，减少内存压力
    # 出现异常值直接并入最大或最小数上
    def statistic_dist(self, x, shap_values):
        # 首先生成特征数
        feature_size = x.shape[1]
        columns = x.columns
        ref_stats_table = {}
        ref_stats = {"feature": [], "mean": [], "std": []}
        for feature_index, feature in enumerate(columns):
            feature_values = x[feature]
            feature_dicts = dict(
                zip(range(self.bin_num), [{"feature_value": [], "shap_value": []} for _ in range(self.bin_num)]))
            for case_index, feature_value in enumerate(feature_values):
                if feature_value <= self.feature_min_values[feature]:
                    feature_dicts[0]["feature_value"].append(feature_value)
                    feature_dicts[0]["shap_value"].append(shap_values[case_index][feature_index])
                elif feature_value >= self.feature_max_values[feature]:
                    feature_dicts[max(feature_dicts.keys())]["feature_value"].append(feature_value)
                    feature_dicts[max(feature_dicts.keys())]["shap_value"].append(
                        shap_values[case_index][feature_index])
                else:
                    key = int((feature_value - self.feature_min_values[feature]) / self.feature_interval[feature])
                    feature_dicts[key]["feature_value"].append(feature_value)
                    feature_dicts[key]["shap_value"].append(shap_values[case_index][feature_index])
            new_feature_dicts = {"bin_num": [], "size": [], "mean": [], "std": []}

            for key in feature_dicts:
                bin_feature_values = feature_dicts[key]["feature_value"]
                bin_shap_values = feature_dicts[key]["shap_value"]
                new_feature_dicts["bin_num"].append(key)
                new_feature_dicts["size"].append(len(bin_feature_values))
                # 恰好该组没有任何的数据,那么mean 和方差赋值为0
                if len(bin_feature_values) == 0:
                    new_feature_dicts["mean"].append(0.0)
                    new_feature_dicts["std"].append(0.0)
                else:
                    new_feature_dicts["mean"].append(np.mean(bin_shap_values))
                    new_feature_dicts["std"].append(np.std(bin_shap_values))

            feature_dataframe = pd.DataFrame(new_feature_dicts)
            ref_stats_table[feature] = feature_dataframe
            ref_stats["feature"].append(feature)
            # 重要bug，写的时候着急了，没怎么确定就直接用了
            ref_stats["mean"].append(np.mean(shap_values[:, feature_index]))
            ref_stats["std"].append(np.std(shap_values[:, feature_index]))
        ref_stats = pd.DataFrame(ref_stats)
        return ref_stats, ref_stats_table

    def feat_selection(self, shap_values):
        feature_important = []
        feature_select = []
        for feature_index, feature in enumerate(self.columns):
            feature_shap_values = shap_values[:, feature_index]
            # 先求绝对值之后再求平均值，作为特征重要性
            feature_important.append(np.mean(abs(feature_shap_values)))
        feature_important = np.array(feature_important) / sum(feature_important)
        feature_important_zip = list(zip(self.columns, feature_important))
        feature_sorted = sorted(feature_important_zip, key=lambda x: x[1], reverse=True)
        print("特征重要性排序：", feature_sorted)
        important_sum = 0
        for feature_name, important in feature_sorted:
            important_sum += important
            feature_select.append(feature_name)
            if important_sum >= self.threshold:
                if len(feature_select) > 20:
                    feature_select = feature_select[:20]
                return feature_select

    def expected_shap_dist(self, ref_states_table, detect_states_table):
        detect_size = detect_states_table["size"]
        ref_mean = ref_states_table["mean"]
        ref_std = ref_states_table["std"]
        length = len(ref_mean)
        expected_mean = 0
        # 先求分子
        for index in range(length):
            expected_mean += ref_mean[index] * detect_size[index]
        # 之后除以分母
        expected_mean = expected_mean / sum(detect_size)

        # 下面开始求期望方差
        # 先从分子开始求
        left_formula = 0
        for index in range(length):
            left_formula += detect_size[index] * (ref_std[index] ** 2)

        right_formula = 0

        for index in range(length):
            right_formula += detect_size[index] * ((ref_mean[index] - expected_mean) ** 2)

        expected_std = ((left_formula + right_formula) / sum(detect_size)) ** 0.5

        return expected_mean, expected_std

    def t_test(self, expected_mean, expected_std, detect_mean, detect_std):
        if math.isnan(expected_std) or math.isnan(expected_mean) or math.isnan(detect_mean) or math.isnan(
                detect_std):
            print("异常值警告!!! , 以下结论存疑:", end=" ")
            return False
        t_value, p_value = stats.ttest_ind_from_stats(expected_mean, expected_std, self.window_size, detect_mean,
                                                      detect_std,
                                                      self.window_size)
        print("当前t值为" + str(t_value), end=" ")
        print("当前双样本p值为" + str(p_value), end=" ")
        if p_value < self.alpha:
            return p_value, True
        else:
            return p_value, False

    def updated_ref_dist_drift(self, ref_stats_table, cur_stats_table):

        return cur_stats_table

    def updated_ref_dist(self, ref_stats_table, cur_stats_table):
        cur_size = cur_stats_table["size"]
        ref_size = ref_stats_table["size"]

        cur_mean = cur_stats_table["mean"]
        ref_mean = ref_stats_table["mean"]

        cur_std = cur_stats_table["std"]
        ref_std = ref_stats_table["std"]

        length = len(ref_stats_table)
        updated_states_table = ref_stats_table.copy(deep=True)

        for index in range(length):
            updated_states_table.loc[index, ["size"]] = cur_size[index] + ref_size[index]

            if updated_states_table["size"][index] == 0:
                updated_states_table.loc[index, ["mean"]] = 0
                updated_states_table.loc[index, ["std"]] = 0
            else:

                updated_states_table.loc[index, ["mean"]] = (ref_mean[index] * ref_size[index] + cur_mean[index] *
                                                             cur_size[index]) / updated_states_table["size"][index]

                left_formula = ((ref_mean[index] - updated_states_table["mean"][index]) ** 2 +
                                ref_std[index] ** 2) * ref_size[index]

                right_formula = ((cur_mean[index] - updated_states_table["mean"][index]) ** 2 +
                                 cur_std[index] ** 2) * cur_size[index]

                updated_states_table.loc[index, ["std"]] = ((left_formula + right_formula) /
                                                            updated_states_table["size"][index]) ** 0.5
        return updated_states_table
