# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf
import glob
import os
import random
import numpy as np

class Dataloader(object):

	def __init__(self, train_dir, num_threads=8, batch_size=8, model_type="valid"):
		"""[summary]

		Arguments:
			object {[type]} -- [description]

		Keyword Arguments:
			num_threads {int} -- [description] (default: {"valid":8, "test":4})
			batch_size {int} -- [description] (default: {"valid":batchsize, "test":1})
			model_type {str} -- [description] (default: {"valid", "test"})
		"""
		self.model_type = model_type

		self.test_filenames= [y for y in glob.glob(os.path.join(train_dir, '*.png'))]
		self.test_filenames.sort()
		self.file_len = len(self.test_filenames)
		print("the number of %s images: %d"%(model_type, self.file_len))

		############################# Dataset构建训练集读取列表 #################################
		test_dataset = tf.data.Dataset.from_tensor_slices(self.test_filenames)
		test_dataset = test_dataset.map(self._parse_function_test, num_parallel_calls=num_threads)
		test_dataset = test_dataset.batch(batch_size)
		self.test_dataset = test_dataset
		self.test_iterator = test_dataset.make_one_shot_iterator()
		self.initializer = self.test_iterator.make_initializer(self.test_dataset)
		self.test_image_batch = self.test_iterator.get_next()


	def _parse_function_test(self, filename):
		"""训练集图片的预处理操作
		Args:
			filename:输入的文件名

		Returns:
			image_decoded: 解码图片
		"""
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_image(image_string, channels=3)

		if self.model_type == "valid6P":
			image_decoded_tf = tf.convert_to_tensor(image_decoded)
			self.h = 256
			image_decoded = tf.concat([image_decoded_tf[:, 0*self.h:1*self.h, :],
									image_decoded_tf[:, 1*self.h:2*self.h, :],
									image_decoded_tf[:, 4*self.h:(4+1)*self.h, :]], axis=1)
		return image_decoded


if __name__ == '__main__':
	pass
