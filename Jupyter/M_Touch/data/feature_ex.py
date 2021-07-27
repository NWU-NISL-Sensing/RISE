# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:13:31 2019
基于手机传感器的身份认证
@author: jiao
"""

import math
from copy import deepcopy

import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, StratifiedKFold
#from sklearn.externals import joblib
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def touch(path,nameall):
    label=[]
    label_num=0;
    name_num=0
    feature_flag=0
    colornum=[]
    for name_num in nameall:
        num_name = 0;
        name_u = name_num

        flag_2 =0
        n1 = []
        x1 = []
        y1 = []
        s1 = []
        t1 = []
        n2 = []
        x2 = []
        y2 = []
        s2 = []
        t2 = []
        n3 = []
        x3 = []
        y3 = []
        s3 = []
        t3 = []
        x_zb1 = []
        y_zb1 = []
        x_zb2 = []
        y_zb2 = []
        x_zb3 = []
        y_zb3 = []
        x_zb4 = []
        y_zb4 = []


        # sign=0 #标记处理文件所处的位置
        # end=len(name_u)
        for num_name in range(len(name_u)):

            filename = name_u[num_name]
            n4 = []
            x4 = []
            y4 = []
            s4 = []
            t4 = []
            with open(path+filename, 'r', encoding='UTF-8') as file_to_read:
                while True:
                    lines = file_to_read.readline()  # 整行读取数据
                    if not lines:
                        break
                        pass
                    n_tmp, x_tmp, y_tmp, s_tmp, t_tmp = [float(i) for i in
                                                         lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。

                    n4.append(n_tmp)  # 添加新读取的数据
                    x4.append(x_tmp)
                    y4.append(y_tmp)
                    s4.append(s_tmp)
                    t4.append(t_tmp)
                    pass
                n4 = np.array(n4)  # 将数据从list类型转换为array类型。
                x4 = np.array(x4)
                y4 = np.array(y4)
                s4 = np.array(s4)
                t4 = np.array(t4)
                pass
            if n1==[]:
                n1 = n4
                x1 = x4
                y1 = y4
                s1 = s4
                t1 = t4
            elif n1!=[] and n2==[]:
                n2 = n4
                x2 = x4
                y2 = y4
                s2 = s4
                t2 = t4
            elif n2!=[] and n3==[]:
                n3 = n4
                x3 = x4
                y3 = y4
                s3 = s4
                t3 = t4

        #处理数据
        i = 0
        j = 0
        flag = 0;
        A = []
        B = []
        C = []
        D = []
        E = []

        color_num=0;
        for inum in range(len(n1)):
            if n1[inum] == 0:
                color_num=color_num+1
        colornum.append(color_num)

        for i in range(len(n1)):
            if flag == 0:
                if n1[i] == 0:
                    if i == 0:
                        xzb1=x1[i]  # x1
                        yzb1=y1[i]  # y1
                        xzb2=x2[i]  # x2
                        yzb2=y2[i]  # y2
                        xzb3=x3[i]  # x3
                        yzb3=y3[i]  # y3
                        xzb4=x4[i]  # x3
                        yzb4=y4[i]  # y3
                        continue;
                    j = i
                    flag = 1
                    A = x1[0:j]
                    A = np.vstack((A, y1[0:j]))
                    A = np.vstack((A, t1[0:j]))
                    A = np.vstack((A, s1[0:j]))
                    B = x2[0:j]
                    B = np.vstack((B, y2[0:j]))
                    B = np.vstack((B, t2[0:j]))
                    B = np.vstack((B, s2[0:j]))
                    C = x3[0:j]
                    C = np.vstack((C, y3[0:j]))
                    C = np.vstack((C, t3[0:j]))
                    C = np.vstack((C, s3[0:j]))
                    D = x4[0:j]
                    D = np.vstack((D, y4[0:j]))
                    D = np.vstack((D, t4[0:j]))
                    D = np.vstack((D, s4[0:j]))


                    #特征计算
                    length1 = 0
                    for x_num in range(len(A[0, :])):
                        if x_num == (len(A[0, :]) - 1):
                            break;
                        length1 = length1 + math.sqrt(
                            (A[0, x_num + 1] - A[0, x_num]) * (A[0, x_num + 1] - A[0, x_num]) + (
                                    A[1, x_num + 1] - A[1, x_num]) * (A[1, x_num + 1] - A[1, x_num]))  # 距离
                    weiyi1 = math.sqrt((A[0, j - 1] - A[0, 0]) * (A[0, j - 1] - A[0, 0]) + (A[1, j - 1] - A[1, 0]) * (
                            A[1, j - 1] - A[1, 0]))  # 位移
                    juli_weiyi1 = length1 / weiyi1  # 距离/位移
                    time1 = A[3, j - 1]  # 持续时间
                    speed1 = length1 / time1  # 平均速度
                    mean1 = np.mean(A[2, :])  # size平均值
                    biaozhuncha1 = np.std(A[2, :], ddof=1)  # size标准差

                    length2 = 0
                    for x_num2 in range(len(B[0, :])):
                        if x_num2 == (len(B[0, :]) - 1):
                            break;
                        length2 = length2 + math.sqrt(
                            (B[0, x_num2 + 1] - B[0, x_num2]) * (B[0, x_num2 + 1] - B[0, x_num2]) + (
                                    B[1, x_num2 + 1] - B[1, x_num2]) * (B[1, x_num2 + 1] - B[1, x_num2]))  # 距离
                    weiyi2 = math.sqrt((B[0, j - 1] - B[0, 0]) * (B[0, j - 1] - B[0, 0]) + (B[1, j - 1] - B[1, 0]) * (
                            B[1, j - 1] - B[1, 0]))  # 位移
                    juli_weiyi2 = length2 / weiyi2  # 距离/位移
                    time2 = B[3, j - 1]  # 持续时间
                    speed2 = length2 / time2  # 平均速度
                    mean2 = np.mean(B[2, :])  # size平均值
                    biaozhuncha2 = np.std(B[2, :], ddof=1)  # size标准差

                    length3 = 0
                    for x_num3 in range(len(C[0, :])):
                        if x_num3 == (len(C[0, :]) - 1):
                            break;
                        length3 = length3 + math.sqrt(
                            (C[0, x_num3 + 1] - C[0, x_num3]) * (C[0, x_num3 + 1] - C[0, x_num3]) + (
                                    C[1, x_num3 + 1] - C[1, x_num3]) * (C[1, x_num3 + 1] - C[1, x_num3]))  # 距离
                    weiyi3 = math.sqrt((C[0, j - 1] - C[0, 0]) * (C[0, j - 1] - C[0, 0]) + (C[1, j - 1] - C[1, 0]) * (
                            C[1, j - 1] - C[1, 0]))  # 位移
                    juli_weiyi3 = length3 / weiyi3  # 距离/位移
                    time3 = C[3, j - 1]  # 持续时间
                    speed3 = length3 / time3  # 平均速度
                    mean3 = np.mean(C[2, :])  # size平均值
                    biaozhuncha3 = np.std(C[2, :], ddof=1)  # size标准差

                    length4 = 0
                    for x_num4 in range(len(C[0, :])):
                        if x_num4 == (len(C[0, :]) - 1):
                            break;
                        length4 = length4 + math.sqrt(
                            (D[0, x_num4 + 1] - D[0, x_num4]) * (D[0, x_num4 + 1] - D[0, x_num4]) + (
                                    D[1, x_num4 + 1] - D[1, x_num4]) * (D[1, x_num4 + 1] - D[1, x_num4]))  # 距离
                    weiyi4 = math.sqrt((D[0, j - 1] - D[0, 0]) * (D[0, j - 1] - D[0, 0]) + (D[1, j - 1] - D[1, 0]) * (
                            D[1, j - 1] - D[1, 0]))  # 位移
                    juli_weiyi4 = length4 / weiyi4  # 距离/位移
                    time4 = D[3, j - 1]  # 持续时间
                    speed4 = length4 / time4  # 平均速度
                    mean4 = np.mean(C[2, :])  # size平均值
                    biaozhuncha4 = np.std(C[2, :], ddof=1)  # size标准差
                    # PY
                    Pf = []
                    Pf.append(xzb1)
                    Pf.append(yzb1)
                    Pf.append(xzb2)
                    Pf.append(yzb2)
                    Pf.append(xzb3)
                    Pf.append(yzb3)
                    Pf.append(xzb4)
                    Pf.append(yzb4)

                    Spf=[]
                    Spf2=[]

                    Spf=sorted([yzb1, yzb2, yzb3, yzb4])
                    for snum in range(4):
                        Spf2.append(Pf[Pf.index(Spf[snum]) - 1])
                        Spf2.append(Spf[snum])
                    Pf=Spf2


                    #
                    d1 = Pf[2] - Pf[0]
                    d2 = Pf[4] - Pf[2]
                    d3 = Pf[6] - Pf[4]
                    d4 = Pf[3] - Pf[1]
                    d5 = Pf[5] - Pf[3]
                    d6 = Pf[7] - Pf[5]
                    d7 = math.sqrt((Pf[2] - Pf[0]) * (Pf[2] - Pf[0]) + (Pf[3] - Pf[1]) * (Pf[3] - Pf[1]))
                    d8 = math.sqrt((Pf[4] - Pf[2]) * (Pf[4] - Pf[2]) + (Pf[5] - Pf[3]) * (Pf[5] - Pf[3]))
                    d9 = math.sqrt((Pf[4] - Pf[0]) * (Pf[4] - Pf[0]) + (Pf[5] - Pf[1]) * (Pf[5] - Pf[1]))
                    d10 = math.sqrt((Pf[6] - Pf[0]) * (Pf[6] - Pf[0]) + (Pf[7] - Pf[1]) * (Pf[7] - Pf[1]))
                    d11 = math.sqrt((Pf[6] - Pf[2]) * (Pf[6] - Pf[2]) + (Pf[7] - Pf[3]) * (Pf[7] - Pf[3]))
                    d12 = math.sqrt((Pf[6] - Pf[4]) * (Pf[6] - Pf[4]) + (Pf[7] - Pf[5]) * (Pf[7] - Pf[5]))

                    #

                    feature = []
                    feature.append((length1+length2+length3+length4)/4)
                    feature.append((weiyi1+weiyi2+weiyi3+weiyi4)/4)
                    feature.append((juli_weiyi1+juli_weiyi2+juli_weiyi3+juli_weiyi4)/4)
                    feature.append((time1+time2+time3+time4)/4)
                    feature.append((speed1+speed2+speed3+speed4)/4)
                    feature.append((mean1+mean2+mean3+mean4)/4)
                    feature.append((biaozhuncha1+biaozhuncha2+biaozhuncha3+biaozhuncha4)/4)
                    feature.append(d1)
                    feature.append(d2)
                    feature.append(d3)
                    feature.append(d4)
                    feature.append(d5)
                    feature.append(d6)
                    feature.append(d7)
                    feature.append(d8)
                    feature.append(d9)
                    feature.append(d10)
                    feature.append(d11)
                    feature.append(d12)
                    E = np.array(feature);

            elif flag == 1:
                if n1[i] == 0:
                    xzb1 = x1[i]  # x1
                    yzb1 = y1[i]  # y1
                    xzb2 = x2[i]  # x2
                    yzb2 = y2[i]  # y2
                    xzb3 = x3[i]  # x3
                    yzb3 = y3[i]  # y3
                    xzb4 = x4[i]  # x3
                    yzb4 = y4[i]  # y3
                    k = j;
                    j = i

                    A = x1[k:j]
                    A = np.vstack((A, y1[k:j]))
                    A = np.vstack((A, t1[k:j]))
                    A = np.vstack((A, s1[k:j]))
                    B = x2[k:j]
                    B = np.vstack((B, y2[k:j]))
                    B = np.vstack((B, t2[k:j]))
                    B = np.vstack((B, s2[k:j]))
                    C = x3[k:j]
                    C = np.vstack((C, y3[k:j]))
                    C = np.vstack((C, t3[k:j]))
                    C = np.vstack((C, s3[k:j]))
                    D = x4[k:j]
                    D = np.vstack((D, y4[k:j]))
                    D = np.vstack((D, t4[k:j]))
                    D = np.vstack((D, s4[k:j]))

                    # 特征计算
                    length1 = 0
                    for x_num1 in range(len(A[0, :])):
                        if x_num1 == (len(A[0, :]) - 1):
                            break;
                        length1 = length1 + math.sqrt(
                            (A[0, x_num1 + 1] - A[0, x_num1]) * (A[0, x_num1 + 1] - A[0, x_num1]) + (
                                    A[1, x_num1 + 1] - A[1, x_num1]) * (A[1, x_num1 + 1] - A[1, x_num1]))  # 距离
                    weiyi1 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                            A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
                    juli_weiyi1 = length1 / weiyi1  # 距离/位移
                    time1 = A[3, len(A) - 1]  # 持续时间
                    speed1 = length1 / time1  # 平均速度
                    mean1 = np.mean(A[2, :])  # size平均值
                    biaozhuncha1 = np.std(A[2, :], ddof=1)  # size标准差

                    length2 = 0
                    for x_num2 in range(len(A[0, :])):
                        if x_num2 == (len(A[0, :]) - 1):
                            break;
                        length2 = length2 + math.sqrt(
                            (A[0, x_num2 + 1] - A[0, x_num2]) * (A[0, x_num2 + 1] - A[0, x_num2]) + (
                                    A[1, x_num2 + 1] - A[1, x_num2]) * (A[1, x_num2 + 1] - A[1, x_num2]))  # 距离
                    weiyi2 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                            A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
                    juli_weiyi2 = length2 / weiyi2  # 距离/位移
                    time2 = A[3, len(A) - 1]  # 持续时间
                    speed2 = length2 / time2  # 平均速度
                    mean2 = np.mean(A[2, :])  # size平均值
                    biaozhuncha2 = np.std(A[2, :], ddof=1)  # size标准差

                    length3 = 0
                    for x_num3 in range(len(A[0, :])):
                        if x_num3 == (len(A[0, :]) - 1):
                            break;
                        length3 = length3 + math.sqrt(
                            (A[0, x_num3 + 1] - A[0, x_num3]) * (A[0, x_num3 + 1] - A[0, x_num3]) + (
                                    A[1, x_num3 + 1] - A[1, x_num3]) * (A[1, x_num3 + 1] - A[1, x_num3]))  # 距离
                    weiyi3 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                            A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
                    juli_weiyi3 = length3 / weiyi3  # 距离/位移
                    time3 = A[3, len(A) - 1]  # 持续时间
                    speed3 = length3 / time3  # 平均速度
                    mean = np.mean(A[2, :])  # size平均值
                    biaozhuncha3 = np.std(A[2, :], ddof=1)  # size标准差

                    length4 = 0
                    for x_num4 in range(len(A[0, :])):
                        if x_num4 == (len(A[0, :]) - 1):
                            break;
                        length4 = length4 + math.sqrt(
                            (A[0, x_num4 + 1] - A[0, x_num4]) * (A[0, x_num4 + 1] - A[0, x_num4]) + (
                                    A[1, x_num4 + 1] - A[1, x_num4]) * (A[1, x_num4 + 1] - A[1, x_num4]))  # 距离
                    weiyi4 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                            A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
                    juli_weiyi4 = length4 / weiyi4  # 距离/位移
                    time4 = A[3, len(A) - 1]  # 持续时间
                    speed4 = length4 / time4  # 平均速度
                    mean = np.mean(A[2, :])  # size平均值
                    biaozhuncha4 = np.std(A[2, :], ddof=1)  # size标准差
                    # PY
                    Pf = []
                    Pf.append(xzb1)
                    Pf.append(yzb1)
                    Pf.append(xzb2)
                    Pf.append(yzb2)
                    Pf.append(xzb3)
                    Pf.append(yzb3)
                    Pf.append(xzb4)
                    Pf.append(yzb4)
                    Spf=[]
                    Spf2=[]

                    Spf=sorted([yzb1, yzb2, yzb3, yzb4])
                    for snum in range(4):
                        Spf2.append(Pf[Pf.index(Spf[snum]) - 1])
                        Spf2.append(Spf[snum])
                    Pf=Spf2
                    #
                    d1 = Pf[2] - Pf[0]
                    d2 = Pf[4] - Pf[2]
                    d3 = Pf[6] - Pf[4]
                    d4 = Pf[3] - Pf[1]
                    d5 = Pf[5] - Pf[3]
                    d6 = Pf[7] - Pf[5]
                    d7 = math.sqrt((Pf[2] - Pf[0]) * (Pf[2] - Pf[0]) + (Pf[3] - Pf[1]) * (Pf[3] - Pf[1]))
                    d8 = math.sqrt((Pf[4] - Pf[2]) * (Pf[4] - Pf[2]) + (Pf[5] - Pf[3]) * (Pf[5] - Pf[3]))
                    d9 = math.sqrt((Pf[4] - Pf[0]) * (Pf[4] - Pf[0]) + (Pf[5] - Pf[1]) * (Pf[5] - Pf[1]))
                    d10 = math.sqrt((Pf[6] - Pf[0]) * (Pf[6] - Pf[0]) + (Pf[7] - Pf[1]) * (Pf[7] - Pf[1]))
                    d11 = math.sqrt((Pf[6] - Pf[2]) * (Pf[6] - Pf[2]) + (Pf[7] - Pf[3]) * (Pf[7] - Pf[3]))
                    d12 = math.sqrt((Pf[6] - Pf[4]) * (Pf[6] - Pf[4]) + (Pf[7] - Pf[5]) * (Pf[7] - Pf[5]))
                    feature = []
                    feature.append((length1 + length2 + length3 + length4) / 4)
                    feature.append((weiyi1 + weiyi2 + weiyi3 + weiyi4) / 4)
                    feature.append((juli_weiyi1 + juli_weiyi2 + juli_weiyi3 + juli_weiyi4) / 4)
                    feature.append((time1 + time2 + time3 + time4) / 4)
                    feature.append((speed1 + speed2 + speed3 + speed4) / 4)
                    feature.append((mean1 + mean2 + mean3 + mean4) / 4)
                    feature.append((biaozhuncha1 + biaozhuncha2 + biaozhuncha3 + biaozhuncha4) / 4)
                    feature.append(d1)
                    feature.append(d2)
                    feature.append(d3)
                    feature.append(d4)
                    feature.append(d5)
                    feature.append(d6)
                    feature.append(d7)
                    feature.append(d8)
                    feature.append(d9)
                    feature.append(d10)
                    feature.append(d11)
                    feature.append(d12)

                    E = np.vstack((E, feature))

            #

        xzb1 = x1[i]  # x1
        yzb1 = y1[i]  # y1
        xzb2 = x2[i]  # x2
        yzb2 = y2[i]  # y2
        xzb3 = x3[i]  # x3
        yzb3 = y3[i]  # y3
        xzb4 = x4[i]  # x3
        yzb4 = y4[i]  # y3
        k = j;
        j = i

        A = x1[k:j]
        A = np.vstack((A, y1[k:j]))
        A = np.vstack((A, t1[k:j]))
        A = np.vstack((A, s1[k:j]))
        B = x2[k:j]
        B = np.vstack((B, y2[k:j]))
        B = np.vstack((B, t2[k:j]))
        B = np.vstack((B, s2[k:j]))
        C = x3[k:j]
        C = np.vstack((C, y3[k:j]))
        C = np.vstack((C, t3[k:j]))
        C = np.vstack((C, s3[k:j]))
        D = x4[k:j]
        D = np.vstack((D, y4[k:j]))
        D = np.vstack((D, t4[k:j]))
        D = np.vstack((D, s4[k:j]))

        # 特征计算
        length1 = 0
        for x_num1 in range(len(A[0, :])):
            if x_num1 == (len(A[0, :]) - 1):
                break;
            length1 = length1 + math.sqrt(
                (A[0, x_num1 + 1] - A[0, x_num1]) * (A[0, x_num1 + 1] - A[0, x_num1]) + (
                        A[1, x_num1 + 1] - A[1, x_num1]) * (A[1, x_num1 + 1] - A[1, x_num1]))  # 距离
        weiyi1 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
        juli_weiyi1 = length1 / weiyi1  # 距离/位移
        time1 = A[3, len(A) - 1]  # 持续时间
        speed1 = length1 / time1  # 平均速度
        mean1 = np.mean(A[2, :])  # size平均值
        biaozhuncha1 = np.std(A[2, :], ddof=1)  # size标准差

        length2 = 0
        for x_num2 in range(len(A[0, :])):
            if x_num2 == (len(A[0, :]) - 1):
                break;
            length2 = length2 + math.sqrt(
                (A[0, x_num2 + 1] - A[0, x_num2]) * (A[0, x_num2 + 1] - A[0, x_num2]) + (
                        A[1, x_num2 + 1] - A[1, x_num2]) * (A[1, x_num2 + 1] - A[1, x_num2]))  # 距离
        weiyi2 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
        juli_weiyi2 = length2 / weiyi2  # 距离/位移
        time2 = A[3, len(A) - 1]  # 持续时间
        speed2 = length2 / time2  # 平均速度
        mean2 = np.mean(A[2, :])  # size平均值
        biaozhuncha2 = np.std(A[2, :], ddof=1)  # size标准差

        length3 = 0
        for x_num3 in range(len(A[0, :])):
            if x_num3 == (len(A[0, :]) - 1):
                break;
            length3 = length3 + math.sqrt(
                (A[0, x_num3 + 1] - A[0, x_num3]) * (A[0, x_num3 + 1] - A[0, x_num3]) + (
                        A[1, x_num3 + 1] - A[1, x_num3]) * (A[1, x_num3 + 1] - A[1, x_num3]))  # 距离
        weiyi3 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
        juli_weiyi3 = length3 / weiyi3  # 距离/位移
        time3 = A[3, len(A) - 1]  # 持续时间
        speed3 = length3 / time3  # 平均速度
        mean = np.mean(A[2, :])  # size平均值
        biaozhuncha3 = np.std(A[2, :], ddof=1)  # size标准差

        length4 = 0
        for x_num4 in range(len(A[0, :])):
            if x_num4 == (len(A[0, :]) - 1):
                break;
            length4 = length4 + math.sqrt(
                (A[0, x_num4 + 1] - A[0, x_num4]) * (A[0, x_num4 + 1] - A[0, x_num4]) + (
                        A[1, x_num4 + 1] - A[1, x_num4]) * (A[1, x_num4 + 1] - A[1, x_num4]))  # 距离
        weiyi4 = math.sqrt((A[0, len(A[0, :]) - 1] - A[0, 0]) * (A[0, len(A[0, :]) - 1] - A[0, 0]) + (
                A[1, len(A[0, :]) - 1] - A[1, 0]) * (A[1, len(A[0, :]) - 1] - A[1, 0]))  # 位移
        juli_weiyi4 = length4 / weiyi4  # 距离/位移
        time4 = A[3, len(A) - 1]  # 持续时间
        speed4 = length4 / time4  # 平均速度
        mean = np.mean(A[2, :])  # size平均值
        biaozhuncha4 = np.std(A[2, :], ddof=1)  # size标准差
        # PY
        Pf = []
        Pf.append(xzb1)
        Pf.append(yzb1)
        Pf.append(xzb2)
        Pf.append(yzb2)
        Pf.append(xzb3)
        Pf.append(yzb3)
        Pf.append(xzb4)
        Pf.append(yzb4)

        Spf = []
        Spf2 = []
        Spf = sorted([yzb1, yzb2, yzb3, yzb4])
        for snum in range(4):
            Spf2.append(Pf[Pf.index(Spf[snum]) - 1])
            Spf2.append(Spf[snum])
        Pf = Spf2
        #
        d1 = Pf[2] - Pf[0]
        d2 = Pf[4] - Pf[2]
        d3 = Pf[6] - Pf[4]
        d4 = Pf[3] - Pf[1]
        d5 = Pf[5] - Pf[3]
        d6 = Pf[7] - Pf[5]
        d7 = math.sqrt((Pf[2] - Pf[0]) * (Pf[2] - Pf[0]) + (Pf[3] - Pf[1]) * (Pf[3] - Pf[1]))
        d8 = math.sqrt((Pf[4] - Pf[2]) * (Pf[4] - Pf[2]) + (Pf[5] - Pf[3]) * (Pf[5] - Pf[3]))
        d9 = math.sqrt((Pf[4] - Pf[0]) * (Pf[4] - Pf[0]) + (Pf[5] - Pf[1]) * (Pf[5] - Pf[1]))
        d10 = math.sqrt((Pf[6] - Pf[0]) * (Pf[6] - Pf[0]) + (Pf[7] - Pf[1]) * (Pf[7] - Pf[1]))
        d11 = math.sqrt((Pf[6] - Pf[2]) * (Pf[6] - Pf[2]) + (Pf[7] - Pf[3]) * (Pf[7] - Pf[3]))
        d12 = math.sqrt((Pf[6] - Pf[4]) * (Pf[6] - Pf[4]) + (Pf[7] - Pf[5]) * (Pf[7] - Pf[5]))
        feature = []
        feature.append((length1 + length2 + length3 + length4) / 4)
        feature.append((weiyi1 + weiyi2 + weiyi3 + weiyi4) / 4)
        feature.append((juli_weiyi1 + juli_weiyi2 + juli_weiyi3 + juli_weiyi4) / 4)
        feature.append((time1 + time2 + time3 + time4) / 4)
        feature.append((speed1 + speed2 + speed3 + speed4) / 4)
        feature.append((mean1 + mean2 + mean3 + mean4) / 4)
        feature.append((biaozhuncha1 + biaozhuncha2 + biaozhuncha3 + biaozhuncha4) / 4)
        feature.append(d1)
        feature.append(d2)
        feature.append(d3)
        feature.append(d4)
        feature.append(d5)
        feature.append(d6)
        feature.append(d7)
        feature.append(d8)
        feature.append(d9)
        feature.append(d10)
        feature.append(d11)
        feature.append(d12)

        E = np.vstack((E, feature)) # 之前叫B

        if flag_2 == 0:
            Feature = E;
            flag_2 = 1;
        elif flag_2 == 1:
            Feature = np.vstack((E, Feature))  # BF

        if feature_flag == 0:
            FeatureAll = Feature
            feature_flag = 1
        elif feature_flag == 1:
            FeatureAll = np.vstack((FeatureAll, Feature))

    #label
        for labelnum in range(len(Feature[:, 0])):
            label.append(label_num)
        label_num = label_num + 1;
    return FeatureAll,label




