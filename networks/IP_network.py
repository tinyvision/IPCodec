# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import tensorflow as tf

from IP_config import PConfig
from networks.modules.baseED.baseE import Encoder
from networks.modules.baseED.baseD import Decoder
from networks.modules.hyperED.hyperE import HyperEncoder
from networks.modules.hyperED.hyperD import HyperDecoder
from networks.modules.hyperED.hyperT import HyperTransform
from networks.modules.hyperED.context import ContextModel
from networks.modules.hyperED.entropy_para import EntropyPara
from networks.modules.baseED.baseU import Unet

from utils.utils import print_tensor, pad, crop, normalize


class PNet(object):

	def __init__(self, is_train=True):

		self.is_train = is_train
		self.encoder = Encoder(is_train, ch_E=PConfig.ch_E[0], is_multi=PConfig.is_multi, loss_metric=PConfig.loss_metric)
		self.decoder = Decoder(is_train, ch_D=PConfig.ch_D[1], is_multi=PConfig.is_multi)
		self.hyper_encoder = HyperEncoder(is_train, ch_E=PConfig.ch_E[1])
		self.hyper_decoder = HyperDecoder(is_train, up_type="deconv", ch_D=PConfig.ch_D[0])
		self.context_model = ContextModel(is_train)
		self.entropy_para = EntropyPara(is_train)
		
		# 增加
		self.hyperResE = HyperEncoder(is_train, name="Pmod_ResE", ch_E=PConfig.ch_E[1])
		self.hyperResD = HyperDecoder(is_train, name="Pmod_ResD", up_type="deconv", ch_D=PConfig.ch_D[0])
		if PConfig.with_context_model: self.context_Res = ContextModel(is_train, name="Pmod_mask")
		self.entropy_Res = EntropyPara(is_train, name="Pmod_entropy", out_filter=PConfig.ch_E[0][-1]*2) # out_fiter is specific for P network.
		self.hyper_trans = HyperTransform(is_train, name="Pmod_trans")

		self.unet = Unet(is_train)


	def __call__(self, pre_img, cur_img, lambda_onehot=None):
		# quantize inputs
		self.image_shape = tf.shape(cur_img)
		# print(self.image_shape.name)
		if PConfig.is_multi:
			self.lambda_onehot = tf.expand_dims(lambda_onehot, 0)
			# print(self.lambda_onehot)
		if not self.is_train: # 当非训练时会对测试图片做pad
			pre_img = pad(pre_img, align=64)
			cur_img = pad(cur_img, align=64)

		#################### Analysis net 产生压缩特征  ############################
		self.y = self.encoder(cur_img, lambda_onehot=self.lambda_onehot) # 提取feature

		self.y_round = tf.stop_gradient(tf.round(self.y) - self.y) + self.y
		
		#################### Analysis_prior net 概率模型熵估计  ##################
		self.z = self.hyper_encoder(self.y_round)  # encode_z编码
		self.z_round = tf.stop_gradient(tf.round(self.z) - self.z) + self.z
		self.z_noise = self.z + tf.random_uniform(tf.shape(self.z), minval=-0.5, maxval=0.5) 
		#训练概率密度估计噪声模拟round的操作
		self.z_hat = self.z_noise if self.is_train else self.z_round

		#################### synthesis_prior net 产生初步概率模型  #################
		self.z_decoder = self.hyper_decoder(self.z_hat)

		#################### context_mask_net 产生概率模型  ########################
		self.y_noise = self.y + tf.random_uniform(tf.shape(self.y), \
										minval=-0.5, maxval=0.5) #训练概率密度估计噪声模拟round的操作
		self.y_hat = self.y_noise if self.is_train else self.y_round
		self.mask_feature = self.context_model(self.y_round)

		self.entropy_para_output = tf.concat([self.mask_feature, self.z_decoder], 3, name='entropy_para_input')

		#################### entropy_para_net 增加3层1*1卷积   ####################
		self.recon_sigma = self.entropy_para(self.entropy_para_output)

		#################### synthesis net 恢复图像  ##############################
		self.recon_image = self.decoder(self.y_round, lambda_onehot=self.lambda_onehot)
		if PConfig.is_post:
			self.recon_image = self.unet(self.recon_image)

		if not self.is_train: # 当非训练时会对测试图片做crop
			self.recon_image = crop(self.recon_image, self.image_shape[1], self.image_shape[2], 64)

		#################### 构建y_res残差hyper熵估计网络  ##############################
		self.y_pre = self.encoder(pre_img, lambda_onehot=self.lambda_onehot) # 提取feature
		self.y_pre_round = tf.stop_gradient(tf.round(self.y_pre) - self.y_pre) + self.y_pre
		self.y_res_round = self.y_round - self.y_pre_round
		#################### 构建y_res残差hyper熵估计z网络  ##############################
		self.z_res = self.hyperResE(tf.concat([self.y_round, self.y_pre_round], -1))
		self.z_res_round = tf.stop_gradient(tf.round(self.z_res) - self.z_res) + self.z_res
		self.z_res_noise = self.z_res + tf.random_uniform(tf.shape(self.z_res), minval=-0.5, maxval=0.5) 
		self.z_res_hat = self.z_res_noise if self.is_train else self.z_res_round
		self.z_res_decoder = self.hyperResD(self.z_res_hat)
		#################### 构建y_res残差hyper熵估计mask网络  ##############################
		self.mask_y_pre = self.hyper_trans(self.y_pre_round)
		if PConfig.with_context_model: 
			self.mask_res = self.context_Res(self.y_res_round)
			self.entropy_para_res = tf.concat([self.mask_res, self.mask_y_pre, self.z_res_decoder], 3, name='entropy_res_input')
		else:
			self.entropy_para_res = tf.concat([self.mask_y_pre, self.z_res_decoder], 3, name='entropy_res_input')
		self.res_sigma = self.entropy_Res(self.entropy_para_res)
		
		return self.recon_image
