#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:22
# @Author  : ZWP
# @Desc    : 代码问题，其中有不少的算法逻辑是通过列名索引的，因此算法要求输入的X变量必须为dataframe的格式
# @File    : main.py
import random
import numpy as np
from model_shap import object_model
from draw import drift_visualization
from drift_detect import drift_detect


# 随机数固定
def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)


def L_Code():
    # 初始化相关参数
    window_size = 1000
    test_size = 0.9
    shap_class = 0
    dataset = "SEA"
    alpha = 0.001
    threshold = 0.99

    # 构造可解释模型以及漂移检测器
    shap_model = object_model(window_size, test_size, shap_class=shap_class, dataset=dataset, external=True)
    detector = drift_detect(X_train=shap_model.X_middle_train, window_size=window_size, alpha=alpha,
                            threshold=threshold)

    # 进行特征的过滤
    X_train, shap = shap_model.getTrain()
    feature_select = detector.feat_selection(shap)

    # 初始化绘图函数
    vis = drift_visualization(feature_select, alpha)

    # 获取第一个时间窗口得值
    end, X_ref, shap_values, is_drift, acc = shap_model.getNextWindows()
    ref_stats, ref_stats_table = detector.statistic_dist(X_ref, shap_values)

    # 循环检测数据流中的漂移
    while True:
        # 获取最新窗口数据
        end, X_detect, shap_detect_values, is_drift, acc = shap_model.getNextWindows()
        if end:
            break
        detect_stats, detect_stats_table = detector.statistic_dist(X_detect, shap_detect_values)

        # 当前检测窗口的窗口号
        window_number_detect = shap_model.windows_number - 1

        # 将数据保存到绘图类中
        vis.add_acc(window_number_detect, acc)

        if is_drift:
            # 上一次取的数据的窗口号
            vis.add_drift(window_number_detect)

        # 设置一下索引列为特证名，后面好找
        temp_detect_stats = detect_stats.set_index("feature")
        temp_ref_stats = ref_stats.set_index("feature")

        # 标识当前窗口是否检测到漂移
        windows_drift = False

        # 对每一个特征进行循环，找到漂移
        for feature_index, feature in enumerate(feature_select):
            expected_mean, expected_std = detector.expected_shap_dist(ref_stats_table[feature],
                                                                      detect_stats_table[feature])

            print("\t 参考窗口统计值：", temp_ref_stats.loc[feature]["mean"], temp_ref_stats.loc[feature]["std"])
            print("\t 期望数据统计值", expected_mean, expected_std)
            print("\t 检测窗口统计值", temp_detect_stats.loc[feature]["mean"], temp_detect_stats.loc[feature]["std"])

            p_value, drift_warning = detector.t_test(expected_mean, expected_std,
                                                     temp_detect_stats.loc[feature]["mean"],
                                                     temp_detect_stats.loc[feature]["std"])
            # 标识当前特征是否检测到漂移
            if drift_warning:
                print(str(feature) + "列发生了漂移")
                windows_drift = True
            else:
                print(str(feature) + "列未发生漂移")

            vis.add_p_value(window_number_detect, feature, p_value, drift_warning)
        # 如果检测到漂移就简单更新，未检测到就积累更新
        if windows_drift:
            X_retrain, X_retrain_shap = shap_model.reTrain(window_number_detect)
            ref_stats, ref_stats_table = detector.statistic_dist(X_retrain, X_retrain_shap)
            # for feature in feature_select:
            #     ref_stats_table[feature] = detector.updated_ref_dist_drift(ref_stats_table[feature],
            #                                                                detect_stats_table[feature])
        else:
            # 对单个特征的表格更新操作
            for feature in feature_select:
                ref_stats_table[feature] = detector.updated_ref_dist(ref_stats_table[feature],
                                                                     detect_stats_table[feature])
    vis.do_draw(feature_select[0])


def L_Code_Plus(window_size, test_size, shap_class, dataset, alpha, threshold):
    # 构造可解释模型以及漂移检测器
    shap_model = object_model(window_size, test_size, shap_class=shap_class, dataset=dataset, external=True)
    detector = drift_detect(X_train=shap_model.X_middle_train, window_size=window_size, alpha=alpha,
                            threshold=threshold)

    # 进行特征的过滤
    X_train, shap = shap_model.getTrain()
    feature_select = detector.feat_selection(shap)

    # 初始化绘图函数
    vis = drift_visualization(feature_select, alpha)

    # 获取第一个时间窗口得值
    end, X_ref, shap_values, is_drift, all_acc = shap_model.getNextWindows()
    ref_stats, ref_stats_table = detector.statistic_dist(X_ref, shap_values)

    # 循环检测数据流中的漂移
    while True:
        # 获取最新窗口数据
        end, X_detect, shap_detect_values, is_drift, all_acc = shap_model.getNextWindows()
        if end:
            break
        detect_stats, detect_stats_table = detector.statistic_dist(X_detect, shap_detect_values)

        # 当前检测窗口的窗口号
        window_number_detect = shap_model.windows_number - 1

        # 将数据保存到绘图类中
        vis.add_acc(window_number_detect, all_acc[0])
        vis.add_ex_acc(window_number_detect, all_acc[1])

        if is_drift:
            # 上一次取的数据的窗口号
            vis.add_drift(window_number_detect)

        # 设置一下索引列为特证名，后面好找
        temp_detect_stats = detect_stats.set_index("feature")
        temp_ref_stats = ref_stats.set_index("feature")

        # 标识当前窗口是否检测到漂移
        windows_drift = False

        # 对每一个特征进行循环，找到漂移
        for feature_index, feature in enumerate(feature_select):
            expected_mean, expected_std = detector.expected_shap_dist(ref_stats_table[feature],
                                                                      detect_stats_table[feature])

            print("\t 参考窗口统计值：", temp_ref_stats.loc[feature]["mean"], temp_ref_stats.loc[feature]["std"])
            print("\t 期望数据统计值", expected_mean, expected_std)
            print("\t 检测窗口统计值", temp_detect_stats.loc[feature]["mean"], temp_detect_stats.loc[feature]["std"])

            p_value, drift_warning = detector.t_test(expected_mean, expected_std,
                                                     temp_detect_stats.loc[feature]["mean"],
                                                     temp_detect_stats.loc[feature]["std"])
            # 标识当前特征是否检测到漂移
            if drift_warning:
                print(str(feature) + "列发生了漂移")
                windows_drift = True
            else:
                print(str(feature) + "列未发生漂移")

            vis.add_p_value(window_number_detect, feature, p_value, drift_warning)
        # 如果检测到漂移就简单更新，未检测到就积累更新
        if windows_drift:
            X_retrain, X_retrain_shap = shap_model.reTrain(window_number_detect)
            ref_stats, ref_stats_table = detector.statistic_dist(X_retrain, X_retrain_shap)
            # for feature in feature_select:
            #     ref_stats_table[feature] = detector.updated_ref_dist_drift(ref_stats_table[feature],
            #                                                                detect_stats_table[feature])
        else:
            pass
            # 对单个特征的表格更新操作
            for feature in feature_select:
                ref_stats_table[feature] = detector.updated_ref_dist(ref_stats_table[feature],
                                                                     detect_stats_table[feature])
    vis.do_draw(feature_select[0])
    return vis.getScore(2)


if __name__ == '__main__':
    # L_Code()
    all_TP = 0
    all_TN = 0
    all_FP = 0
    all_FN = 0
    window_size = 600
    test_size = 0.6
    shap_class = 0
    dataset = "phishing"
    alpha = 0.01
    threshold = 0.99
    for i in range(1):
        TP, TN, FP, FN = L_Code_Plus(window_size, test_size, shap_class, dataset, alpha, threshold)
        all_TP += TP
        all_TN += TN
        all_FP += FP
        all_FN += FN
    TPR = all_TP / (all_TP + all_FN)
    FPR = all_FP / (all_FP + all_TN)
    precision = all_TP / (all_TP + all_FP)
    recall = all_TP / (all_TP + all_FN)
    accuracy = (all_TP + all_TN) / (all_TP + all_TN + all_FN + all_FP)
    print("多次测试的结果：", all_TP, all_TN, all_FP, all_FN)

    print("TPR:", TPR)
    print("FPR:", FPR)
    print("precision:", precision)
    print("recall:", recall)
    print("accuracy:", accuracy)
