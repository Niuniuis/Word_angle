#-*- coding:utf-8 -*-
from keras.applications.vgg16 import preprocess_input,VGG16
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
from keras.optimizers import SGD
import time
import cv2
from math import *
from scipy.stats import mode
import tensorflow as tf
import os
from keras.layers import Dense
from keras.models import Model

# g1 = tf.Graph()
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# sess1 = tf.Session(config=config,graph=g1)
#
# ##########################################################################################################################
# """
# 原始的AngleModle
# """
#
#
# def get_session(gpu_fraction=1.0):
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#
# def load_model():
#     with sess1.as_default():
#         with g1.as_default():
#             sgd = SGD(lr=0.00001, momentum=0.9)
#             model = VGG16(weights=None, classes=4)
#             # 加载模型权重
#             model.load_weights('./models/modelAngle.h5', by_name=True)
#             # 编译模型，以较小的学习率进行训练
#             model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#             return model
#
# # 加载模型
# # K.set_session(get_session(0.5))
# # K.set_session(sess1)
# # graph = tf.get_default_graph()
# model = load_model()
# def predict(im):
#     """
#     图片文字方向预测
#     """
#     with sess1.as_default():
#         with sess1.graph.as_default():
#             ROTATE = [0, 270, 180, 90]
#             # im = cv2.imread(img)
#             w, h ,_= im.shape
#             thesh = 0.07
#             xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
#             #im = im.crop((xmin, ymin, xmax, ymax))  # 剪切图片边缘，清楚边缘噪声
#             im = im[xmin:xmax, ymin:ymax]
#             # im = im.resize((224, 224))
#             im = cv2.resize(im, (224, 224))
#             img = np.array(im)
#             img = preprocess_input(img.astype(np.float32))
#             pred = model.predict(np.array([img]))
#             index = np.argmax(pred, axis=1)[0]
#     return ROTATE[index]

def rotate(image, angle, center=None, scale=1.0): #1
    filled_color = -1
    if filled_color == -1:
        filled_color = mode([image[0, 0], image[0, -1],
                             image[-1, 0], image[-1, -1]]).mode[0]
    if np.array(filled_color).shape[0] == 2:
        if isinstance(filled_color, int):
            filled_color = (filled_color, filled_color, filled_color)
    else:
        filled_color = tuple([int(i) for i in filled_color])

    (h, w) = image.shape[:2] #2
    height, width = image.shape[:2]
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))  # 这个公式参考之前内容
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)  # 逆时针旋转 degree

    matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=filled_color)
    # imgRotation = cv2.warpAffine(transform_img, matRotation, (widthNew, heightNew), borderVal

    return imgRotation #7

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

############################################################################################################################
"""
chineseocr_Anglemoudle: 效果略胜于原有的AngleModle
"""

g2 = tf.Graph()
AngleModelPb = "models\\Angle-model.pb"
AngleModelPbtxt = "models\\Angle-model.pbtxt"
opencvFlag = "keras"

def get_sessions():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess2 = tf.Session(config=config,graph=g2)
    with sess2.as_default():
        with g2.as_default():
            with gfile.FastGFile(AngleModelPb, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            inputImg = g2.get_tensor_by_name('input_1:0')
            predictions = g2.get_tensor_by_name('predictions/Softmax:0')
            keep_prob = tf.placeholder(tf.float32)
    return sess2,inputImg,predictions,keep_prob

sess2,inputImg,predictions,keep_prob = get_sessions()

def angle_detect_tf(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].astype(np.float32)

    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img = np.array([img])
    with sess2.as_default():
        with sess2.graph.as_default():
            out = sess2.run(predictions, feed_dict={inputImg: img,
                                                   keep_prob: 0
                                                   })

    index = np.argmax(out, axis=1)[0]
    return ROTATE[index]


def angle_detect(img, adjust=True):
    """
    文字方向检测
    """
    if opencvFlag == 'keras':
        return angle_detect_tf(img, adjust=adjust)


#########################################################################################################################
# """
# 百度云下载的AngleModle
# """
# g3 = tf.Graph()
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# sess3 = tf.Session(config=config, graph=g3)
#
# def load_sess3():
#     with sess3.as_default():
#         with g3.as_default():
#             vgg = VGG16(weights=None, input_shape=(224, 224, 3))
#             # 修改输出层 3个输出
#             x = vgg.layers[-2].output
#             predictions_class = Dense(
#                 4, activation='softmax', name='predictions_class')(x)
#             prediction = [predictions_class]
#             model = Model(inputs=vgg.input, outputs=prediction)
#             sgd = SGD(lr=0.00001, momentum=0.9)
#             model.compile(
#                 optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#             model.load_weights(
#                 'E:\\Text_data\\Angle_Modle\\modelAngle.h5')
#     return model
#
#
# # 加载模型
# model3 = load_sess3()
#
# def predict_sess3(path=None,img=None):
#     """
#     图片文字方向预测
#     """
#     ROTATE = [0, 90, 180, 270]
#     if path is not None:
#         im = Image.open(path).convert('RGB')
#     elif img is not None:
#         im = Image.fromarray(img).convert('RGB')
#     w, h = im.size
#     # 对图像进行剪裁
#     # 左上角(int(0.1 * w), int(0.1 * h))
#     # 右下角(w - int(0.1 * w), h - int(0.1 * h))
#     xmin, ymin, xmax, ymax = int(0.1 * w), int(
#         0.1 * h), w - int(0.1 * w), h - int(0.1 * h)
#     im = im.crop((xmin, ymin, xmax, ymax))  # 剪切图片边缘，清除边缘噪声
#     # 对图片进行剪裁之后进行resize成(224,224)
#     im = im.resize((224, 224))
#     # 将图像转化成数组形式
#     img = np.array(im)
#     with sess3.as_default():
#         with sess3.graph.as_default():
#             img = preprocess_input(img.astype(np.float32))
#             pred = model3.predict(np.array([img]))
#             index = np.argmax(pred, axis=1)[0]
#     return ROTATE[index]


if __name__ == "__main__":
    t = time.time()
    img = cv_imread("E:\\工作数据文件夹\\案例\\gexian\\2018042000076\\288e3993edf4468aa2f63a49bd6cbb80.jpg")
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_RGB2GRAY)
    ima_ge = np.expand_dims(gray, axis=2)
    new_gray = np.concatenate((ima_ge, ima_ge, ima_ge), axis=-1)
    # degree = predict(new_gray)
    degree = angle_detect(new_gray)
    # degree2 = predict_sess3(img=new_gray)

    # degree = predict(img)
    # print(degree)
    # transform_img = rotate(img, degree)
    # cv2.imwrite("st2a.png", transform_img)
    # print("旋转角度:"+str(predict("37.jpg"))+"°")
    # print(type(predict("37.jpg")))
    print("旋转检测时间:{:.2f}秒".format(time.time() - t))
