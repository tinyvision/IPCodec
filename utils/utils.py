# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf
import glob
import os
import random
import numpy as np
from PIL import Image

def str2bool(str):
        return True if str.lower() == "true" else False

def print_tensor(t):
    print(t.op.name, ' ', t.get_shape().as_list())


class imageUtil:
    """根据batch加载numpy的图片
    Args:
        test_set_dir: 测试集或者验证集的地址

    Returns:
        返回各种图片
    """
    def __init__(self, test_set_dir = './test_set/'):
        self.testFileNames=os.listdir(test_set_dir)
        self.testFileNames.sort()
        self.imageSetNumber = len(self.testFileNames)
        self.test_set_dir = test_set_dir
        self.testImgId = 0
        self.testImgSetId = 0
        self.updateTestImg()

    # 获得文件夹中图片的数量
    def getTestSetSize(self):
        return self.imageSetNumber
    
    # 获得测试集文件的名称
    def getTestName(self, test_num):
        return self.testFileNames[test_num]

    def updateTestImg(self):
        with Image.open(self.test_set_dir+(self.testFileNames[self.testImgSetId])) as img:
        # img = Image.open(self.test_set_dir+(self.testFileNames[self.testImgSetId]))
            self.test_image_array = np.array(img)#.astype(np.float32)
        # print(self.test_set_dir+(self.testFileNames[self.testImgSetId]))
        self.test_image_float_array = self.test_image_array# / 255.0
        self.testImgId = (self.testImgId + 1) % (self.imageSetNumber)
        self.testImgSetId = self.testImgId % self.imageSetNumber
    
    # 获得单张图片的numpy集合
    def getOneTestImage(self):
        npArray = self.test_image_float_array
        while len(npArray.shape) != 3: #or (npArray[0:10, 0:10, 0] - npArray[0:10, 0:10, 1] < 1e-3).all():
            self.updateTestImg()
            # print('here', len(npArray.shape))
            npArray = self.test_image_float_array
        npArray = npArray[np.newaxis, ...]
        self.updateTestImg()
        return npArray

    # 制造测试集不同batch大小的图像集合
    def generateTestImageBatch(self, batch_size, is_train=True):
        name_batch = self.testFileNames[self.testImgSetId]
        tensor_batch = self.getOneTestImage()
        image_name_batch = list()
        image_name_batch.append(name_batch)
        for i in range(batch_size-1):
            name_batch = self.testFileNames[self.testImgSetId]
            tensor_batch = np.concatenate((tensor_batch, self.getOneTestImage()), axis=0)
            image_name_batch.append(name_batch)
        return tensor_batch, image_name_batch


def load_single_image(filename, scale=255.0, is_zero_one=True, dtype=np.uint32):
    """加载单张图片为 numpy 数组
    Args:
        filename: 图片文件名
        scale: 输入图片像素的范围
        dtype: 数据类型

    Returns:
        np.ndarray: 图片数组。形状为 (1, height, width, 3)
    """
    if is_zero_one:
        array = np.array(Image.open(filename), dtype=dtype)#/ 255
    else:
        array = np.array(Image.open(filename), dtype=dtype) / (scale / 2) - 1
    return np.expand_dims(array[..., :3], 0)


def aligned_length(length, align, is_tensor=True):
    if is_tensor:
        return tf.cast(tf.ceil(length / align), tf.int32) * align
    return int(np.ceil(length / align) * align)


def pad(image, align=64):
    """将图片 padding 对齐为指定大小的倍数
    Args:
        image: 输入图片 np.ndarray
        align: pad 对齐的倍数

    Returns:
        np.ndarray: pad 后的图片
    """

    def _pad(length):
        padding = aligned_length(length, align) - length
        p0 = padding // 2
        p1 = padding - p0
        return p0, p1

    image_shape = tf.shape(image)
    padding = ((0, 0), _pad(image_shape[1]), _pad(image_shape[2]), (0, 0))
    return tf.pad(image, padding, 'CONSTANT')


def crop(image, height, width, align):
    """将图片 crop 为 pad 之前的大小
    Args:
        image: 输入图片 tensor
        height: 图片原高度 tensor
        width:  图片原宽度 tensor
        align:  pad 对齐的倍数

    Returns:
        tensor: crop 后的图片
    """

    def _offset(length):
        return (aligned_length(length, align, True) - length) // 2

    crop_box = _offset(height), _offset(width), height, width
    cropped_image = tf.image.crop_to_bounding_box(image, *crop_box)
    # cropped_image.set_shape(image.shape)  # 去除 crop 操作后的 shape 中实际上可确定的 None

    return cropped_image

def normalize(image, scale=255.0, dtype=tf.uint8, name=None):
    """将图片 normalize 为指定的 scale 和 dtype
    Args:
        image: 输入图片 tensor
        scale: 输出的 scale，默认为 255.0
        dtype: 输出的数据类型，默认为 tf.uint8
        name: 操作的名称

    Returns:
        tensor: normalize 后的图片
    """
    # clipped = tf.clip_by_value(image, 0.0, 1)
    normalized = tf.cast(tf.round(tf.clip_by_value(image * 255, 0, scale)), dtype, name)
    return normalized


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads
    

def generate_weight_list(list_index, list_ratio):
    new_list = []
    for ii in range(len(list_index)):
        for jj in range(list_ratio[ii]):
            new_list.append(list_index[ii])
    # print(new_list)
    return new_list
