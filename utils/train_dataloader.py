# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf
import glob
import os
import random
import numpy as np

class InputDataloader(object):

	def __init__(self, train_dir, h, w, num_threads=16, batch_size=8, model_type="P"):
		"""[summary]

		Arguments:
			object {[type]} -- [description]

		Keyword Arguments:
			num_threads {int} -- [description] (default: {16})
			batch_size {int} -- [description] (default: {8})
			model_type {str} -- [description] (default: {"6P", "P", "I"})
		"""
		self.model_type = model_type
		self.h = h
		self.w = w
		train_filenames= [y for y in glob.glob(os.path.join(train_dir, '*.png'))]
		random.shuffle(train_filenames)
		self.file_len = len(train_filenames)
		print("the number of train images: %d"%(self.file_len))
		############################# Dataset构建训练集读取列表 #################################
		train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
		train_dataset = train_dataset.map(self._parse_function_train, num_parallel_calls=num_threads)
		train_dataset = train_dataset.shuffle(buffer_size=50*batch_size).repeat().batch(batch_size).prefetch(num_threads)
		train_iterator = train_dataset.make_one_shot_iterator()
		self.train_image_batch = train_iterator.get_next()


	def _parse_function_train(self, filename):
		"""训练集图片的预处理操作
		Args:
			filename:输入的文件名

		Returns:
			image_normed: 剪切处理后的图片
		"""
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_image(image_string, channels=3)
		if self.model_type == "I":
			image_decoded = tf.random_crop(image_decoded, [self.h, self.w , 3])
			image_decoded = tf.image.random_flip_left_right(image_decoded)
			
		image_decoded = tf.image.random_flip_up_down(image_decoded)

		if self.model_type == "6P":
			# Input image consists of 1I1Rec6P. Randomly select 1P from 6P to simulate the big movement
			image_decoded_tf = tf.convert_to_tensor(image_decoded)
			p_idx = tf.random_uniform([], minval=1, maxval=8, dtype=tf.int32) 
			image_decoded = tf.concat([image_decoded_tf[:, 0*self.h:1*self.h, :],
									image_decoded_tf[:, 1*self.h:2*self.h, :],
									image_decoded_tf[:, p_idx*self.h:(p_idx+1)*self.h, :]], axis=1)
		image_decoded.set_shape((self.h, self.w, 3))

		return image_decoded



if __name__ == '__main__':
	pass
