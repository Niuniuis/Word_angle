
# -*- coding: utf-8 -*-
import os,cv2
import numpy as np
import math
from math import *
from scipy.stats import mode
import threading
import time

class ImageProcessing(object):
    """图片归一化"""
    def __init__(self, img_path):
        self.original_img = self.cv_imread(img_path)
        self.img = self.original_img
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
        self.border_img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def rough_handling(self):
        """图片预处理"""
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3, 3),0)
        blurred = cv2.blur(gray, (5, 5))
        ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10) #腐蚀膨胀对白色像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dst = cv2.dilate(binary, kernel)  # 膨胀
        # cv2.imshow("dst",dst)
        # dst1 = cv2.erode(dst,kernel,iterations=2)
        # cv2.imshow("dst1",dst1)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # 腐蚀+膨胀 去噪 实际效果 连线
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) #膨胀 +  腐蚀 连线  实际效果 去噪
        return gray, blurred, binary, closed, opened

    def rough_handling1(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3, 3),0)
        blurred = cv2.blur(gray,(5,5))
        # ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10) #腐蚀膨胀对白色像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dst = cv2.dilate(binary, kernel)  # 膨胀
        # cv2.imshow("dst",dst)
        # dst1 = cv2.erode(dst,kernel,iterations=2)
        # cv2.imshow("dst1",dst1)
        closed = cv2.dilate(binary, kernel)
        # closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 腐蚀+膨胀 去噪 实际效果 连线
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 膨胀+腐蚀 连线  实际效果 去噪
        # opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # 腐蚀+膨胀 去噪 实际效果 连线
        # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) #膨胀+腐蚀 连线 实际效果 去噪
        return gray, blurred, binary, closed, opened

    def cv_imread(self,file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    def findcontours(self, handling_img):
        """查找纸张轮廓"""
        contour_img = self.img.copy()
        cnts = cv2.findContours(handling_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        docCnt = None
        # 确保至少找到一个轮廓
        if len(cnts) > 0:
            # 按轮廓大小降序排列
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                # 近似轮廓
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                flag = False
                # 如果我们的近似轮廓有四个点，则确定找到了纸
                if len(approx) == 4 and np.min(approx) != 0:
                    if cv2.contourArea(approx) < 0.3 * self.h_scale * self.w_scale:
                        continue
                    docCnt = approx.tolist()
                    cv2.line(contour_img, tuple(docCnt[1][0]), tuple(docCnt[2][0]), (0, 255, 0), 3)
                    cv2.line(contour_img, tuple(docCnt[1][0]), tuple(docCnt[0][0]), (0, 255, 0), 3)
                    cv2.line(contour_img, tuple(docCnt[2][0]), tuple(docCnt[3][0]), (0, 255, 0), 3)
                    cv2.line(contour_img, tuple(docCnt[0][0]), tuple(docCnt[3][0]), (0, 255, 0), 3)
                    docCnt = sorted(docCnt, key=lambda p: p[0][0])
                    docCnt = [[docCnt[0][0]], [docCnt[1][0]], [docCnt[2][0]], [docCnt[3][0]]] if docCnt[0][0][1] - docCnt[1][0][1] < 0 else [[docCnt[1][0]], [docCnt[0][0]], [docCnt[2][0]], [docCnt[3][0]]]
                    docCnt = [[docCnt[0][0]], [docCnt[1][0]], [docCnt[2][0]], [docCnt[3][0]]] if docCnt[2][0][1] - docCnt[3][0][1] > 0 else [[docCnt[0][0]], [docCnt[1][0]], [docCnt[3][0]], [docCnt[2][0]]]
                    docCnt1 = [round(i / self.scale) for x in docCnt for b in x for i in b]
                    original_docCnt = [[[docCnt1[0],docCnt1[1]]],[[docCnt1[2],docCnt1[3]]],[[docCnt1[4],docCnt1[5]]],[[docCnt1[6],docCnt1[7]]]]
                    flag = True
                    break
            if flag :
                print("Find Suited Contour")
                return contour_img, docCnt,original_docCnt
            else:
                print("Don't Find Suited Contour")
                return None
        else:
            print("Don't Find Contour")
            return None

    def mser_text(self,gray):
        """
            :param gray: 灰度图
            :return: keep，文本框坐标
            :return: img_mser，轮廓图
            :return: vis，文本框图
            :return: img_box，文本框图
        """
        #mser 提取文本区域
        img_mser = self.border_img.copy()
        # ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 20)
        # cv2.imshow("mser_gray", gray)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        mser = cv2.MSER_create(_min_area=100, _max_area=500, _max_variation=0.9)  # 得到mser算法对象
        # mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)  # 获取文本区域
        # print(regions)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
        cv2.polylines(img_mser, hulls, 1, (0, 255, 0))

        img_box = self.border_img.copy()

        # for box in _:
        #     x, y, w, h = box
        #     # if abs(w-h)<5: continue
        #     # if w/h < 0.35 or h/w < 0.35: continue
        #     # if w<10 or h<10:
        #     #     continue
        #     cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #化成文本区域轮廓
        vis = self.border_img.copy()
        keep = []
        for c in hulls:
            x, y, w, h = cv2.boundingRect(c)
            if abs(w - h) < 5: continue
                # if x>200 and x< self.w-200:
                #     continue
                # if y>200 and y<self.h-200:
                #     continue
            # min_rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(min_rect)
            # box = np.int0(box)
            # cv2.line(vis, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
            # cv2.line(vis, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
            # cv2.line(vis, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
            # cv2.line(vis, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
            if w / h < 0.35 or h / w < 0.35: continue
            if w < 10 or h < 10: #最小阈值
                continue
            keep.append([x, y, x + w, y + h])
            # cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        return keep, img_mser, vis, img_box #imgbox 和vis 一样

    def non_max_suppression_fast(self, boxes, overlapThresh):
        """

        :param boxes: 文本框坐标
        :param overlapThresh: 重叠阈值
        :return: 合格框
        """
        # 空数组检测
        if len(boxes) == 0:
            return []

            # 将类型转为float
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        # grab the coordinates of the bounding boxes
        # 四个坐标数组
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算面积数组
        idxs = np.argsort(y2)  # 返回的是右下角坐标从小到大的索引值

        # 开始遍历删除重复的框
        while len(idxs) > 0:
            # 将最右下方的框放入pick数组
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # 找到剩下的其余框中最大的坐标x1y1，和最小的坐标x2y2,
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # 计算重叠面积占对应框的比例
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            # 如果占比大于阈值，则删除
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def py_cpu_nms(self,dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]  # bbox打分

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 打分从大到小排列，取index
        order = scores.argsort()[::-1]
        # keep为最后保留的边框
        keep = []
        while order.size > 0:
            # order[0]是当前分数最大的窗口，肯定保留
            i = order[0]
            keep.append(i)
            # 计算窗口i与其他所有窗口的交叠部分的面积
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 交/并得到iou值
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
            inds = np.where(ovr <= thresh)[0]
            # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
            order = order[inds + 1]

        return keep

    #不完整检测
    def screen_box(self, box, box_max_h, error_value, box_number): #pick, 100, 8, 5
        """
        :param:box, box_max_h, error_value, box_number:边框坐标。坐标高度最大值，误差值，文本框数量
        :return: draw_img, new_screen_box,img_transform_box, original_img_transform_box, original_img_transform_boundingrect_box
        归一化图，填充图片后的坐标，透视变换坐标，对应原图变换坐标，原图上的外接矩形坐标
        """
        # 判断文本框是否超出边界
        new_screen_box = []
        box_left = []
        box_right = []
        box_up = []
        box_down = []
        # 超出边界的框
        print(self.h_scale)
        for (startX, startY, endX, endY) in box:
            # if endY - startY >= box_max_h:
            #     continue
            if startX < 50 + error_value:
                box_left.append([startX, startY, endX, endY])
            if endX > self.w_scale + 50 - error_value:
                box_right.append([startX, startY, endX, endY])
            if startY < 50 + error_value:
                box_up.append([startX, startY, endX, endY])
            if endY > self.h_scale + 50 - error_value:
                # print([[startX, startY, endX, endY]])
                box_down.append([startX, startY, endX, endY])
            new_screen_box.append([startX, startY, endX, endY])

        temp_area = 40 #边缘部分 文本框阈值  面积？
        #判断图片左边区域的文本框
        box_left_n = 0  # 计数过小框的个数
        for i in box_left:
            if (i[3] - i[1]) * (i[2] - i[0]) < temp_area:
                box_left_n += 1
                if i in new_screen_box:
                    new_screen_box.remove(i)
            # print(box_left_n)
        if len(box_left) - box_left_n > box_number:
            print("Image left beyond", len(box_left))
            return {"code":"11033","msg":"图片左侧拍摄不完整，或者纸张左侧外有文字，或者背景太复杂"}
        else:
            if len(box_left) > 0:
                for b in box_left:
                    if b in new_screen_box:
                        new_screen_box.remove(b)
        # 判断图片右边区域的文本框
        box_right_n = 0 #计数过小框的个数
        # print("box_right:",box_right)
        for i in box_right:
            # print(i)
            if (i[3] - i[1]) * (i[2] - i[0]) < temp_area:
                box_right_n +=1
                if i in new_screen_box:
                    new_screen_box.remove(i)
            # print("box_right_n:",box_right_n)
        # print(len(box_right) - box_right_n, box_number)
        if len(box_right)- box_right_n > box_number:
            print("Image right beyond", len(box_right))
            # for i in box_right:
            #     print(i,"area:",(i[3] - i[1]) * (i[2] - i[0]))
            return {"code":"11034","msg": "图片右侧拍摄不完整，或者纸张右侧外有文字，或者背景太复杂"}
        else:
            if len(box_right) > 0:
                for b in box_right:
                    if b in new_screen_box:
                        new_screen_box.remove(b)
        # 判断图片上边区域的文本框
        box_up_n = 0  # 计数过小框的个数
        for i in box_up:
            if (i[3] - i[1]) * (i[2] - i[0]) < temp_area:
                box_up_n += 1
                if i in new_screen_box:
                    new_screen_box.remove(i)
        if len(box_up) - box_up_n > box_number:
            print("Image up beyond", len(box_up))
            return {"code":"11035", "msg":"图片上边缘拍摄不完整，或者纸张上边缘外有文字，或者背景太复杂"}
        else:
            if len(box_up) > 0:
                for b in box_up:
                    if b in new_screen_box:
                        new_screen_box.remove(b)
        # 判断图片下边区域的文本框
        box_down_n = 0  # 计数过小框的个数
        for i in box_down:
            # print(i)
            if (i[3] - i[1]) * (i[2] - i[0]) < temp_area:
                box_down_n += 1
                if i in new_screen_box:
                    new_screen_box.remove(i)
        if len(box_down) - box_down_n > box_number:
            print("Image down beyond", len(box_down))
            return {"code":"11036", "msg":"图片下边缘拍摄不完整，或者纸张下边缘外有文字，或者背景太复杂"}
        else:
            if len(box_down) > 0:
                for b in box_down:
                    if b in new_screen_box:
                        new_screen_box.remove(b)

        new_screen_box = np.array(new_screen_box)

        # 查找上下左右 最值点
        min_up = 100000
        min_left = 100000
        max_down = 0
        max_right = 0
        for (startX, startY, endX, endY) in new_screen_box:
            if startX < min_left:
                pos_left = (startX, startY)
                min_left = startX
            if startY < min_up:
                pos_up = (startX, startY)
                min_up = startY
            if endX > max_right:
                pos_right = (endX, endY)
                max_right = endX
            if endY > max_down:
                pos_down = (endX, endY)
                max_down = endY

        # print(pos_up[1],pos_down[1],pos_left[0],pos_right[0])
        # img_split = self.img.copy()[pos_up[1]:pos_down[1],pos_left[0]:pos_right[0]]
        # cv2.imshow("img_split", img_split)

        if len(new_screen_box) == 0:
            print("筛选结束后无合格文本框")
            return {"code":"11037", "msg":"非文本图片"}
        cnt = []
        for (startX, startY, endX, endY) in new_screen_box:
            # 10,最小外接矩形太接近文字
            # cnt.append([[startX - 10, startY - 10]])
            # cnt.append([[startX - 10, endY + 10]])
            # cnt.append([[endX + 10, endY + 10]])
            # cnt.append([[endX + 10, startY - 10]])
            cnt.append([[startX - 50 - 10, startY - 50 - 10]])
            cnt.append([[startX - 50 - 10, endY - 50 + 0]])
            cnt.append([[endX - 50 + 10, endY - 50 + 10]])
            cnt.append([[endX - 50 + 10, startY - 50 - 10]])

        #外接矩形
        x, y, w, h = cv2.boundingRect(np.array(cnt))
        original_boundingrect = self.original_img.copy()
        x = round(x / self.scale) if x >= 0 else 0
        y = round(y / self.scale) if y >= 0 else 0
        w = round(w / self.scale) if x + w <= self.w else abs(x - self.w)
        h = round(h / self.scale) if y + h <= self.h else abs(y - self.h)
        cv2.rectangle(original_boundingrect, (x, y), (x + w, y + h), (255, 255, 0), 3)
        # cv2.namedWindow("original_boundingrect",cv2.WINDOW_NORMAL)
        # cv2.imshow("original_boundingrect", original_boundingrect)
        original_img_transform_boundingrect_box = [[[x,y]],[[x,y+h]],[[x+w,y+h]],[[x+w,y]]]

        rect = cv2.minAreaRect(np.array(cnt))
        box = cv2.boxPoints(rect)
        box1 = np.int0(box)
        box = sorted(box, key=lambda b: b[0])
        box = [box[0],box[1],box[2],box[3]] if box[0][1] < box[1][1] else [box[1],box[0],box[2],box[3]]
        # 换结构
        img_transform_box = [[box[0]],[box[1]],[box[2]],[box[3]]] if box[2][1] > box[3][1] else [[box[0]],[box[1]],[box[3]],[box[2]]]
        # print(img_transform_box)
        img_transform_box1 = [round(i / self.scale) for x in img_transform_box for b in x for i in b]
        # print(img_transform_box1)
        original_img_transform_box = [[[img_transform_box1[0],img_transform_box1[1]]],[[img_transform_box1[2],img_transform_box1[3]]],[[img_transform_box1[4],img_transform_box1[5]]],[[img_transform_box1[6],img_transform_box1[7]]]]
        img_minarearect = self.img.copy()
        draw_img = cv2.drawContours(img_minarearect, [box1], -1, (0, 0, 255), 3)
        #draw_img 归一化图
        # new_screen_box 填充图片后的坐标
        return draw_img, new_screen_box,img_transform_box, original_img_transform_box, original_img_transform_boundingrect_box

