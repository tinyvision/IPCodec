# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import tensorflow as tf
gdn = tf.contrib.layers.gdn

from networks.modules.ops import lambda_mask, print_tensor, gdn_adjust


class Encoder(object):

	def __init__(self, is_train=True, name="encoder", ch_E=[192, 192, 192, 256], is_multi=False, loss_metric="PSNR"):

		self.is_train = is_train
		self.name = name
		self.ch_E = ch_E
		self.is_multi = is_multi
		self.loss_metric = loss_metric


	def __call__(self, input_image, kernel_size=5, lambda_onehot=None):

		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			flow = input_image

			with tf.variable_scope('down0'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[0], kernel_size=(5,4), strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_0")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask0")
					
				flow = gdn(flow)
				# print_tensor(flow)

			with tf.variable_scope('down1'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[1], kernel_size=(4,5), strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_1")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask1")				
				flow = gdn(flow)
				# print_tensor(flow)

			with tf.variable_scope('down2'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[2], kernel_size=(5,4), strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_2")
				
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask2")

				# SSIM模型在做tensorRT变换的时候会出现数值溢出，需要对数值做缩放
				if "SSIM" in self.loss_metric and self.is_train==False:
					flow = flow/100.0
					flow = gdn_adjust(flow, coeff=1e4)
				else:
					flow = gdn(flow)
				# print_tensor(flow)

			with tf.variable_scope('down3'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[3], kernel_size=(4,5), strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_3")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask3")				
				# print_tensor(flow)

		return flow