# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import tensorflow as tf

from networks.modules.ops import lambda_mask, print_tensor


class Unet(object):

	def __init__(self, is_train=True, name="post_unet", is_multi=False):

		self.is_train = is_train
		self.name = name
		self.is_multi = is_multi


	def conv_pool(self, flow, filters_1, filters_2, kernel_size, name = 'conv2d'):
		with tf.variable_scope(name):
			conv_1 = tf.layers.conv2d(inputs=flow, filters=filters_1, kernel_size=kernel_size, padding="same",
										activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name = "conv_1")
			conv_2 = tf.layers.conv2d(inputs=conv_1, filters=filters_2, kernel_size=kernel_size, padding="same",
										activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name = "conv_2")
			pool = tf.layers.max_pooling2d(inputs = conv_2, pool_size = [2,2], strides = 2, padding = "same", name = 'pool')
		return conv_2, pool


	def upconv_concat(self, flow1, flow2, filters, kernel_size, name="upconv"):
		with tf.variable_scope(name):
			flow = tf.layers.conv2d_transpose(inputs=flow1, filters=filters, kernel_size=kernel_size, strides = (2,2), padding ="same",
											activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name = 'up_conv')
		return tf.concat([flow, flow2], axis=-1, name="concat")


	def __call__(self, input_image, kernel_size=5, lambda_onehot=None):

		with tf.variable_scope(self.name):
			################根据npy文件导入学习的权重参数，提取特征#################
			res_weights_np = np.load('./networks/modules/baseED/res_weight_np.npy')
			kernelSize = np.shape(res_weights_np)
			weights_init = tf.constant_initializer(res_weights_np, dtype=tf.float32)
			res_weights = tf.get_variable(name='res_extract_layer_weights', shape=kernelSize, initializer=weights_init,
										trainable=False, dtype=tf.float32)
			input_feature = tf.nn.conv2d(input_image, res_weights, [1, 1, 1, 1], 'SAME', name='res_extract_layer')
			################根据npy文件导入学习的权重参数，提取特征#################

			# tf.summary.image('input_feature', input_feature[..., 0:1], 1, family='Post')
			input_image_new = tf.concat([input_image, input_feature], axis=-1)

			conv_1, pool_1 = self.conv_pool(input_image_new, 64, 64, [3,3], name="conv_pool_1")
			conv_2, pool_2 = self.conv_pool(pool_1, 128, 128, [3,3], name="conv_pool_2")
			conv_3, pool_3 = self.conv_pool(pool_2, 256, 256, [3,3], name="conv_pool_3")
			conv_4, pool_4 = self.conv_pool(pool_3, 512, 512, [3,3], name="conv_pool_4")
			conv_5, pool_5 = self.conv_pool(pool_4, 1024, 1024, [3,3], name="conv_pool_5")
			upconv_6 = self.upconv_concat(conv_5, conv_4, 512, [2,2], name="upconv_6")
			conv_6, pool_6 = self.conv_pool(upconv_6, 512, 512, [3,3], name="conv_pool_6")
			upconv_7 = self.upconv_concat(conv_6, conv_3, 256, [2,2], name="upconv_7")
			conv_7, pool_7 = self.conv_pool(upconv_7, 256, 256, [3,3], name="conv_pool_7")
			upconv_8 = self.upconv_concat(conv_7, conv_2, 128, [2,2], name="upconv_8")
			conv_8, pool_8 = self.conv_pool(upconv_8, 128, 128, [3,3], name="conv_pool_8")
			upconv_9 = self.upconv_concat(conv_8, conv_1, 64, [2,2], name="upconv_9")
			conv_9, pool_9 = self.conv_pool(upconv_9, 64, 64, [3,3], name="conv_pool_9")
			recon_image = tf.layers.conv2d(inputs=conv_9, filters=3, kernel_size=[3,3],padding="same",
								activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "recon_image")

			return recon_image + input_image