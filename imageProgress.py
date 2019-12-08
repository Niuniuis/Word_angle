# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
import math
from math import *
from scipy.stats import mode
import threading
import time
from imageDetection import *
# from models import *
from models2 import *

import traceback
import logging

logging.basicConfig(filename='Img_Progress_log.log',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)

def imageTransform(img, point):
    """

    :param img: 待透视变换图片
    :param point: 透视的坐标
    :return:透视变换后图片
    """
    # print(np.array(point).tolist())
    # print(point)
    """透视变换"""
    # point 坐标 左上 、左下、右下 、右上
    x1, x2, x3, x4 = point[0][0][0], point[1][0][0], point[2][0][0], point[3][0][0]
    y1, y2, y3, y4 = point[0][0][1], point[1][0][1], point[2][0][1], point[3][0][1]
    # imgW = abs(x1 - x4) if abs(x1 - x4) < abs(x2 - x3) else abs(x2 - x3)
    # # print(imgW)
    # imgH = abs(y1 - y2) if abs(y1 - y2) < abs(y4 - y3) else abs(y4 - y3)
    img_up = int(math.sqrt(np.square(x1 - x4) + np.square(y1 - y4)))
    img_down = int(math.sqrt(np.square(x2 - x3) + np.square(y2 - y3)))
    img_left = int(math.sqrt(np.square(x1 - x2) + np.square(y1 - y2)))
    img_right = int(math.sqrt(np.square(x4 - x3) + np.square(y4 - y3)))
    imgW = img_up if img_up > img_down else img_down
    imgH = img_left if img_left > img_right else img_right
    # imgW, imgH = int(imgW), int(imgH)
    pts1 = np.float32(point)
    # cv2.imshow("pts1", pts1)
    # pts2 = np.float32(point)
    pts2 = np.float32([[0, 0], [0, imgH], [imgW, imgH], [imgW, 0]])
    # pts2 = np.float32([[0, 0],  [0, 1000], [1000, 1000],[1000, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (imgW, imgH))
    return dst

class ImgCorrect():
    """图片归一化"""
    def __init__(self, transform_img):
        self.img = transform_img
        self.h, self.w, self.channel = self.img.shape
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.w_scale = 700
            self.h_scale = self.h * self.scale
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.h_scale = 700
            self.w_scale = self.w * self.scale
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)

    #轮廓直线检测 @XYL
    def img_lines(self):
        """
        :return: 画上直线图片
        """
        # hough 提取直线
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("bin",binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 矩形结构
        binary = cv2.dilate(binary, kernel)  # 膨胀
        edges = cv2.Canny(binary, 50, 200)
        # cv2.imshow("edges", edges)
        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # 提取为二维
        # print(lines1)
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return imglines

    # 轮廓直线检测 @YK
    def img_lines1(self):
        """
        基于文本块的形态检测边缘轮廓后检测直线
        :return:边缘轮廓图像
        """
        # ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        # cv2.imshow("binary",binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        binary = cv2.dilate(binary, kernel, iterations=2)  # 膨胀
        edges1 = np.zeros_like(binary)
        images_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            #画上轮廓 去除靠边上的轮廓(疑似黑边)
            cont_T = cont.T
            max_x, min_x = np.max(cont_T[0]), np.min(cont_T[0])
            max_y, min_y = np.max(cont_T[1]), np.min(cont_T[1])
            if min_x < 5 or max_x > self.w - 5: continue
            if min_y < 5 or max_y > self.h - 5: continue
            cv2.drawContours(edges1, [cont], -1, (250, 255, 255), 2)
        # cv2.imshow("Lines",edges1)
        # cv2.waitKey()
        self.lines = cv2.HoughLinesP(edges1, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        if self.lines is None:
            print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # 提取为二维
        # print(lines1)
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return imglines

    #获取图像的倾斜角度传统算法 @YK
    def get_angle(self):
        """
        计算检测到直线角度
        :return: 出现次数最多的角度，即图像角度
        """
        lines = self.lines[:, 0, :]
        """
        clos: 竖线收集的角度，收集容错5°的所有角度及数量求均值
        rows: 横线线收集的角度，收集容错5°的所有角度及数量求均值(根据目前文本分析主要针对水平方向上的线进行处理)
        """
        display = 5  #角度误差
        v = 0
        l = 0
        v_ = []  # 收集正角度
        l_ = []  # 收集负角度
        rows = {0:[0,0]}  ####目前只根据横线进行检测
        clos = {0:[0,0]}
        for i in range(int(90/display)):
            v += display
            l -= display
            rows[v] = [0,0]
            rows[l] = [0,0]
            clos[v] = [0,0]
            clos[l] = [0, 0]
            v_.append(v)
            l_.append(l)
        for x in lines:
            x1, y1, x2, y2 = x
            # 横线
            if abs(y2 - y1) < (x2 - x1):
                if (x2 - x1) == 0:
                    rows[0][0] += 0
                    rows[0][1] += 1
                    continue
                angle = degrees(atan((y2 - y1) / (x2 - x1)))
                angle_last = 0
                for i in range(len(v_)):
                    if angle > 0:
                        if angle_last < angle <= v_[i]:
                            rows[v_[i]][0] += angle
                            rows[v_[i]][1] += 1
                            break
                        angle_last = v_[i]
                        continue
                    elif angle == 0:
                        rows[0][0] += angle
                        rows[0][1] += 1
                        break
                    else:
                        if l_[i] <= angle < angle_last:
                            rows[l_[i]][0] += angle
                            rows[l_[i]][1] += 1
                            break
                        angle_last = v_[i]
                        continue
            else:
                if (y2 - y1) == 0:
                    clos[0][0] += 0
                    clos[0][1] += 1
                    continue
                angle = degrees(atan((y2 - y1) / (x2 - x1)))
                angle_last = 0
                for i in range(len(v_)):
                    if angle > 0:
                        if angle_last < angle <= v_[i]:
                            clos[v_[i]][0] += angle
                            clos[v_[i]][1] += 1
                            break
                        angle_last = v_[i]
                        continue
                    elif angle == 0:
                        clos[0][0] += angle
                        clos[0][1] += 1
                        break
                    else:
                        if l_[i] <= angle < angle_last:
                            clos[l_[i]][0] += angle
                            clos[l_[i]][1] += 1
                            break
                        angle_last = v_[i]
                        continue

        new_rows = sorted(rows.items(), key=lambda item: item[1][1])
        new_clos = sorted(clos.items(), key=lambda item: item[1][1])
        new_rows = new_rows[::-1]
        new_clos = new_clos[::-1]
        try:
            if  new_rows[0][1][1]>new_clos[0][1][1]:
                if new_rows[0][1][1] == 0:
                    return 0
                else:
                    # print(new_rows)
                    # print(new_clos)
                    # print(new_rows[0][1][0] / new_rows[0][1][1])
                    return round(new_rows[0][1][0] / new_rows[0][1][1])
            else:
                if new_clos[0][1][1] == 0:
                    return 0
                else:
                    # print(new_rows)
                    # print(new_clos)
                    # print(new_clos[0][1][0] / new_clos[0][1][1])
                    return round(new_clos[0][1][0] / new_clos[0][1][1])
        except:
            print("--------------------------------------------------")
            print(new_rows)
            print(new_clos)

    # 获取图像的倾斜角度传统算法  @XYL
    def search_lines(self):
        """
        :return: 返回统计后的图片角度
        """
        #对直线进行筛选，确定图片的角度
        lines = self.lines[:, 0, :]  # 提取为二维
        # k = [(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in lines]
        # sorted_k = sorted(lines, key=lambda x:(x[3] - x[1]) / (x[2] - x[0]))
        number_inexistence_k = 0
        sum_positive_k45 = 0
        number_positive_k45 = 0
        sum_positive_k90 = 0
        number_positive_k90 = 0
        sum_negative_k45 = 0
        number_negative_k45 = 0
        sum_negative_k90 = 0
        number_negative_k90 = 0
        sum_zero_k = 0
        number_zero_k = 0
        for x in lines:
            if x[2] == x[0]:
                number_inexistence_k += 1
                continue
            # print(degrees(atan((x[3] - x[1]) / (x[2] - x[0]))),"pos:",x[0],x[1],x[2],x[3],"斜率:",(x[3] - x[1]) / (x[2] - x[0]))
            if 0 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 45:
                number_positive_k45 += 1
                sum_positive_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if 45 <= degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 90:
                number_positive_k90 += 1
                sum_positive_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if -45 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 0:
                number_negative_k45 += 1
                sum_negative_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if -90 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) <= -45:
                number_negative_k90 += 1
                sum_negative_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if x[3] == x[1]:
                number_zero_k += 1

        max_number = max(number_inexistence_k, number_positive_k45, number_positive_k90, number_negative_k45,
                         number_negative_k90, number_zero_k)
        # print(number_inexistence_k,number_positive_k45, number_positive_k90, number_negative_k45, number_negative_k90,number_zero_k)
        if max_number == number_inexistence_k:
            return 90
        if max_number == number_positive_k45:
            return sum_positive_k45 / number_positive_k45
        if max_number == number_positive_k90:
            return sum_positive_k90 / number_positive_k90
        if max_number == number_negative_k45:
            return sum_negative_k45 / number_negative_k45
        if max_number == number_negative_k90:
            return sum_negative_k90 / number_negative_k90
        if max_number == number_zero_k:
            return 0

    #根据角度旋转图像 @XYL
    def rotate_image(self, degree, transform_img):
        """
        正角 逆时针旋转
        :param degree:旋转角度
        :param transform_img:待旋转图片
        :return:旋转后图片
        """
        # print("degree:", degree)
        if -45 <= degree <= 0:
            degree = degree  # #负角度 顺时针
        if -90 <= degree < -45:
            degree = 90 + degree  # 正角度 逆时针
        if 0 < degree <= 45:
            degree = degree  # 正角度 逆时针
        if 45 < degree < 90:
            degree = degree - 90  # 负角度 顺时针
        print("rotate degree:", degree)
        # degree = -45
        # # 获取旋转后4角的填充色
        filled_color = -1
        if filled_color == -1:
            filled_color = mode([transform_img[0, 0], transform_img[0, -1],
                                 transform_img[-1, 0], transform_img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])

        # degree = degree - 90
        height, width = transform_img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # 逆时针旋转 degree

        matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
        matRotation[1, 2] += (heightNew - height) / 2

        imgRotation = cv2.warpAffine(transform_img, matRotation, (widthNew, heightNew), borderValue=filled_color)
        # imgRotation = cv2.warpAffine(transform_img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        # 填充四个角
        # mask = np.zeros((heightNew + 2, widthNew + 2), np.uint8)
        # mask[:] = 0
        # seed_points = [(0, 0), (0, heightNew - 1), (widthNew - 1, 0),
        #                (widthNew - 1, heightNew - 1)]
        # for i in seed_points:
        #     cv2.floodFill(imgRotation, mask, i, filled_color)
        # cv2.namedWindow("imgRotation", cv2.WINDOW_NORMAL)
        # cv2.imshow("imgRotation", imgRotation)
        # cv2.imwrite("./image/yb1.jpg", imgRotation)
        return imgRotation

class SplitImg():
    #根据轮廓分割图片
    def __init__(self, im):
        self.img = im

    def split_img(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        blurred = cv2.blur(gray, (3, 3))
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25,
                                       10)  # 腐蚀膨胀对白色像素
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        dst1 = cv2.erode(binary, kernel1)
        dst2 = cv2.dilate(dst1, kernel2)
        edges = cv2.Canny(dst2, 50, 200)
        cloneimage, contours, heriachy = cv2.findContours(dst2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("cloneimage", cloneimage)
        c = sorted(contours, key=cv2.contourArea, reverse=True)
        little_split_img = []
        for i, contour in enumerate(c):
            if i < 10:
                rect = cv2.minAreaRect(contour)
                # bo = np.array(cv2.boxPoints(rect)).tolist()
                box = np.int0(cv2.boxPoints(rect))
                bo = np.array(box).tolist()
                box1 = sorted(bo, key=lambda b: b[0])
                box1 = [box1[0], box1[1], box1[2], box1[3]] if box1[0][1] < box1[1][1] else [box1[1], box1[0], box1[2],
                                                                                             box1[3]]
                # 换结构
                box1 = [[box1[0]], [box1[1]], [box1[2]], [box1[3]]] if box1[2][1] > box1[3][1] else [[box1[0]],
                                                                                                     [box1[1]],
                                                                                                     [box1[3]],
                                                                                                     [box1[2]]]
                # print(len(little_split_img))
                # print("--:",box1)#[[[29.0, 55.0]], [[29.0, 259.0]], [[422.0, 259.0]], [[422.0, 55.0]]]
                if len(little_split_img) == 0:
                    little_split_img.append(box1)
                    # draw a bounding box arounded the detected barcode and display the image
                    draw_img = cv2.drawContours(self.img.copy(), [box], -1, (0, 0, 255), 3)
                    # cv2.imshow("draw_img" + str(i), draw_img)
                else:
                    temp = True
                    for s in little_split_img:
                        # print(len(little_split_img))
                        # print(min(s[0][0][0], s[1][0][0]),max(s[2][0][0], s[3][0][0]),min(s[0][0][1], s[3][0][1]),max(s[1][0][1], s[2][0][1]))
                        #筛选出四个角点坐标的最值
                        if (min(s[0][0][0], s[1][0][0]) <= box1[0][0][0] <= max(s[2][0][0], s[3][0][0]) and min(
                                s[0][0][1], s[3][0][1]) <= box1[0][0][1] <= max(s[1][0][1], s[2][0][1])) and \
                                (min(s[0][0][0], s[1][0][0]) <= box1[1][0][0] <= max(s[2][0][0], s[3][0][0]) and min(
                                    s[0][0][1], s[3][0][1]) <= box1[1][0][1] <= max(s[1][0][1], s[2][0][1])) and \
                                (min(s[0][0][0], s[1][0][0]) <= box1[2][0][0] <= max(s[2][0][0], s[3][0][0]) and min(
                                    s[0][0][1], s[3][0][1]) <= box1[2][0][1] <= max(s[1][0][1], s[2][0][1])) and \
                                (min(s[0][0][0], s[1][0][0]) <= box1[3][0][0] <= max(s[2][0][0], s[3][0][0]) and min(
                                    s[0][0][1], s[3][0][1]) <= box1[3][0][1] <= max(s[1][0][1], s[2][0][1])):
                            # if min(s[0][0][0],s[1][0][0]) <= box1[0][0][0] <= max(s[2][0][0],s[3][0][0]) and min(s[0][0][1],s[3][0][1]) <= box1[0][0][1] <= max(s[1][0][1],s[2][0][1]) and \
                            #     min(s[0][0][0], s[1][0][0]) <= box1[1][0][0] <= max(s[2][0][0], s[3][0][0]) and min(s[0][0][1], s[3][0][1]) <= box1[1][0][1] <= max(s[1][0][1], s[2][0][1]) and \
                            # min(s[0][0][0], s[1][0][0]) <= box1[2][0][0] <= max(s[3][0][0], s[3][0][0]) and min(s[0][0][1],s[3][0][1]) <= box1[3][0][1] <= max(s[1][0][1], s[2][0][1]) and \
                            #  min(s[0][0][0], s[1][0][0]) <= box1[0][0][0] <= max(s[2][0][0],s[3][0][0]) and min(s[0][0][1], s[3][0][1]) <= box1[0][0][1] <= max(s[1][0][1], s[2][0][1]):
                            temp = True
                            continue
                    if temp == True:
                        # draw a bounding box arounded the detected barcode and display the image
                        draw_img = cv2.drawContours(draw_img, [box], -1, (0, 0, 255), 3)
                        # cv2.imshow("draw_img" + str(i), draw_img)
                        little_split_img.append(box1)

def contours_transform(filename, contours):
    """
    :param filename: 图片路径
    :param contours: 纸张轮廓坐标
    :return: 透视后图片
    """
    #检测图片轮廓
    image_processing = ImageProcessing(filename)
    # img_tran = imageTransform(image_processing.original_img.copy(), contours)
    img_tran = image_processing.original_img.copy()
    return img_tran

def mser_tran_rotate(filename, screen_img):
    """
    :param filename: 图片路径
    :param screen_img: 文本区域轮廓坐标
    :return: 变换后的图片
    """
    image_processing = ImageProcessing(filename)
    img_mser_transform = image_processing.original_img.copy()
    print("img_mser_transform",img_mser_transform.shape)
    ##透视变换
    # img_mser_transform = imageTransform(img_mser_transform, screen_img)


    imgcorrect = ImgCorrect(img_mser_transform)

    # ########################旋转图像 @XYL
    # lines_img = imgcorrect.img_lines()
    #
    # if lines_img is None:
    #     rotate_img = imgcorrect.rotate_image(0, img_mser_transform)
    #     print("rotate image1")
    #     return rotate_img
    # else:
    #     degree = imgcorrect.search_lines()
    #     rotate_img = imgcorrect.rotate_image(degree, img_mser_transform)
    #     print("rotate_image2")
    #     return rotate_img

    ########################旋转图像 @YK
    lines_img = imgcorrect.img_lines1()
    if imgcorrect.lines is None:
        return img_mser_transform
    degree = imgcorrect.get_angle()
    print('---',degree)
    if degree!=0:
        rotate_img = get_img_rot_broa(img_mser_transform,degree=degree)
        return rotate_img
    else:
        return img_mser_transform


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

# 根据角度旋转图像 @YK
def get_img_rot_broa(img, degree=0, filled_color=-1):
        """
        Desciption:
                Get img rotated a certain degree,
            and use some color to fill 4 corners of the new img.
        """

        # 获取旋转后4角的填充色
        if filled_color == -1:
            filled_color = mode([img[0, 0], img[0, -1],
                                 img[-1, 0], img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])

        height, width = img.shape[:2]

        # 旋转后的尺寸
        height_new = int(width * fabs(sin(radians(degree))) +
                         height * fabs(cos(radians(degree))))
        width_new = int(height * fabs(sin(radians(degree))) +
                        width * fabs(cos(radians(degree))))

        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2

        # Pay attention to the type of elements of filler_color, which should be
        # the int in pure python, instead of those in numpy.
        img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                     borderValue=filled_color)
        # 填充四个角
        mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
        mask[:] = 0
        seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
                       (width_new - 1, height_new - 1)]
        for i in seed_points:
            cv2.floodFill(img_rotated, mask, i, filled_color)

        return img_rotated

def detection_main(file_name):
    new_dict = {}
    detection_result = detection_iuiqd(file_name)
    image_ = cv_imread(file_name)
    ###### 测试用代码块
    # degree = predict(image_) #原始文字反向检测模型
    # degree = angle_detect(image_) #chineseocr-master的文字方向检测模型
    # degree2 = predict_sess3(img=image_) #百度云下载的AngleMoudle 准确率在88%

    # new_dict['code'] = detection_result['code']
    # new_dict['degree'] = degree
    # new_dict['file_name'] = file_name
    # detection_result_join = str(new_dict)
    # with open("mark1.txt",'a+',encoding='utf-8') as file:
    #     file.write(detection_result_join+'\n')

    # print("detection_result:", detection_result)
    t_start = time.time()
    if detection_result["code"] == "11040":
        trans_contour_im = contours_transform(detection_result["filename"], detection_result["position"])
        imgcorrect = ImgCorrect(trans_contour_im)
        lines_img = imgcorrect.img_lines()
        if lines_img is None:
            rotate_img = imgcorrect.rotate_image(0, trans_contour_im)
            print("transform rotate image1")
            # degree = predict(rotate_img.copy())
            degree = angle_detect(image_)
            print("predict_degree:", degree)
            transform_img = rotate(rotate_img, degree)
            # cv2.imwrite("./test.jpg", transform_img)
        else:
            degree = imgcorrect.search_lines()
            rotate_img = imgcorrect.rotate_image(degree, trans_contour_im)
            print("transform rotate_image2")
            # cv2.imshow("transform_rotate_image2",rotate_img)
            # degree = predict(rotate_img.copy())
            degree = angle_detect(rotate_img.copy())
            print("predict_degree:", degree)
            transform_img = rotate(rotate_img, degree)
            # cv2.imwrite("./test.jpg", transform_img)
    if detection_result["code"] == "11041":
        im = mser_tran_rotate(detection_result["filename"], detection_result["position"])
        # degree = predict(im.copy())
        degree = angle_detect(im.copy())
        print("predict_degree:", degree)
        transform_img = rotate(im, degree)
    # cv2.imwrite("./test.jpg", transform_img)
    # cv2.imwrite("./image/st2bbbb.png",im)
    # x = np.array(a["image"],np.uint8)
    # cv2.imwrite("./image/gb1a.png", x)
    #
    # # print(type(x))
    # cv2.imshow("x",x)
    # print("耗时:", time.time() - t_start)
    # cv2.waitKey(0)

def detection_main2(file_path):
    t_start = time.time()
    image = cv_imread(file_path)

    imgcorrect = ImgCorrect(image)
    lines_img = imgcorrect.img_lines1()
    degree = imgcorrect.get_angle()
    print("自定义算法检测:",degree)
    if degree!=0:
        rotate_img = get_img_rot_broa(image, degree=degree)
        degree = angle_detect(rotate_img)
        print("模型检测角度:",degree)
        transform_img = get_img_rot_broa(rotate_img, degree=degree)
    else:
        degree = angle_detect(trans_contour_im)
        print("模型检测角度:", degree)
        transform_img = get_img_rot_broa(image, degree=degree)

    print("耗时:", time.time() - t_start)
    return transform_img


if __name__ == "__main__":

    test_path = "images/"

    file_list = os.listdir(test_path)
    for i in file_list:
        if i[-4:] == ".jpg" or i[-4:] == ".png":
            new_path = test_path + i
            print(new_path)
            cv2.namedWindow("image",cv2.WINDOW_NORMAL)
            cv2.namedWindow("img_", cv2.WINDOW_NORMAL)
            image = cv_imread(new_path)
            cv2.imshow("image",image)
            h, w, c = image.shape
            if c != 3:
                cv2.imwrite("new_image.jpg", image)
                image = cv2.imread("new_image.jpg")
                new_path = "new_image.jpg"
            img_ =  detection_main2(new_path)
            cv2.imshow("img_",img_)
            cv2.imwrite("mark_img/%s"%i,img_)
            cv2.waitKey()

