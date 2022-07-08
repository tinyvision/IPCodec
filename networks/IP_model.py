# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import numpy as np
import tensorflow as tf

from IP_config import PConfig
from networks.IP_network import PNet
from networks.modules.entropy.estimator import Entropy

from utils.train_pipeline import InputPipeline
from utils.train_dataloader import InputDataloader
from utils.dataloader import Dataloader
from utils.utils import average_gradients


class PModel(object):
    
    def __init__(self, is_train):

        self.is_train = is_train
        self.total_batch_size = PConfig.total_batch_size if is_train else 1
        self.Entr = Entropy()
        self.Entr_res = Entropy(name="Pmod")
        self.print_opt_var = True

    def build(self):

        with tf.Graph().as_default() as self.graph:
            
            # 使用固话模型的时候可以简单调用graph导入
            if PConfig.use_frozen_model:
                self.import_graph()
            elif self.is_train:
                self.build_dataset()
                self.build_train()
                self.summary()
                self.var_print()
            # 固化时单独固话解码端
            # elif PConfig.freeze_decoder:
            #     self.build_model_decoder()
            else:
                self.build_model()

            # create a session and load the checkpoint
            self.create_session() 
        
        return self.graph, self.sess


    def build_dataset(self):
        
        print("**************** the dataset information ****************")
        print("the train dir is %s" % PConfig.train_set_dir)
        # train_loader = InputDataloader(PConfig.train_set_dir, PConfig.height, PConfig.width, 16, self.total_batch_size, model_type="P") # dataloader的方式
        train_loader = InputPipeline(PConfig.train_set_dir, PConfig.height, PConfig.width, 64, self.total_batch_size, model_type="6P") # pipeline的方式
        self.train_len = train_loader.file_len
        self.train_next = train_loader.train_image_batch

        print("the valid dir is %s" % PConfig.valid_set_dir)
        valid_loader = Dataloader(PConfig.valid_set_dir, 8, self.total_batch_size, model_type="valid6P") # pipeline的方式
        self.valid_len = valid_loader.file_len
        self.init_valid = valid_loader.initializer
        self.valid_next = valid_loader.test_image_batch


    def build_train(self):
        
        self.global_step = tf.train.get_or_create_global_step()
        self.initial_learning_rate = tf.train.piecewise_constant(self.global_step, \
            boundaries=PConfig.boundaries, values=PConfig.multi_learning_rates)
        self.LR = self.initial_learning_rate
        opt = tf.train.AdamOptimizer(self.initial_learning_rate)
        
        if PConfig.gpus_list:
            models = []
            for index, gpu_id in enumerate(PConfig.gpus_list):
                with tf.device('/gpu:%d' % index):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        print('Building Graph on  %dth GPU, id %d' % (index+1, gpu_id))
                        self.build_model()
                        if PConfig.is_multi:
                            models.append((self.input_image_in, self.lambda_onehot, self.grad_and_vars))
                        else:
                            models.append((self.input_image_in, self.grad_and_vars))
            
            if PConfig.is_multi:
                self.tower_input, self.tower_onehot, tower_grads = zip(*models)
            else:
                self.tower_input, tower_grads = zip(*models)
            print(tower_grads)
            self.train_ops = opt.apply_gradients(average_gradients(tower_grads), global_step=self.global_step)

        else:
            with tf.variable_scope(tf.get_variable_scope()):
                print('Graph was built on CPU %d')
                self.build_model()
                self.train_ops = opt.apply_gradients(self.grad_and_vars, global_step=self.global_step)


    def build_model(self):

        with tf.variable_scope('IModel'):

            if not PConfig.freeze_fix_img:
                self.input_image_in = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name="input_image")
            else:
                self.input_image_in = tf.placeholder(dtype=tf.uint8, shape=PConfig.fix_size, name="input_image")
                
            if PConfig.is_multi:
                self.lambda_onehot = tf.placeholder(dtype=tf.float32, shape=[len(PConfig.lambda_list)], name="lambda_onehot")
                self.lambda_train = tf.reduce_sum(tf.multiply(self.lambda_onehot, PConfig.lambda_list))
            else:
                self.lambda_train = tf.cast(tf.convert_to_tensor(PConfig.train_lambda), dtype=tf.float32)
                   
            input_image = tf.cast(self.input_image_in, dtype=tf.float32)
            input_image = tf.div(input_image, 255, name="normalization")
            self.pre_img, self.pre_img_recon, self.cur_img = tf.split(input_image, 3, axis=-2)
            
            self.PNet = PNet(is_train=self.is_train)
            if PConfig.is_multi:
                self.recon_image = self.PNet(self.pre_img, self.cur_img, self.lambda_onehot)
            else:
                self.recon_image = self.PNet(self.pre_img, self.cur_img)
            
                
            self.clip_recon_image = tf.clip_by_value(self.recon_image, 0, 1)

            with tf.name_scope("res_rate"):
                total_bits_y_res = self.Entr_res.entropy_y(self.PNet.y_res_round, self.PNet.res_sigma)
                total_bits_z_res = self.Entr_res.entropy_z(self.PNet.z_res_hat)

                im_shape = tf.cast(tf.shape(self.cur_img), tf.float32)
                self.bpp_y_res = total_bits_y_res / (im_shape[0]  * im_shape[1] * im_shape[2])
                # print(type(total_bits_y), type(self.total_batch_size), type(im_shape[1]), type(im_shape[2]))
                self.bpp_z_res = total_bits_z_res / (im_shape[0]  * im_shape[1] * im_shape[2])
                self.bpp_res = self.bpp_y_res + self.bpp_z_res #the estimatied bit rates
                self.rate_loss = self.bpp_res
            
            with tf.name_scope("rate"):
                total_bits_y = self.Entr.entropy_y(self.PNet.y_hat, self.PNet.recon_sigma)
                total_bits_z = self.Entr.entropy_z(self.PNet.z_hat)

                im_shape = tf.cast(tf.shape(self.cur_img), tf.float32)
                self.bpp_y = total_bits_y / (im_shape[0]  * im_shape[1] * im_shape[2])
                # print(type(total_bits_y), type(self.total_batch_size), type(im_shape[1]), type(im_shape[2]))
                self.bpp_z = total_bits_z / (im_shape[0]  * im_shape[1] * im_shape[2])
                self.bpp = self.bpp_y + self.bpp_z #the estimatied bit rates
                # self.rate_loss = self.bpp
            
            with tf.name_scope("distortion"):
                self.mse_loss = tf.reduce_mean(tf.square((self.cur_img) - (self.recon_image)))
                self.mae_loss = tf.reduce_mean(tf.abs((self.cur_img) - (self.recon_image)))
                self.ms_ssim_d = tf.reduce_mean(tf.image.ssim_multiscale(self.cur_img, self.recon_image, max_val=1.0))

                self.cur_img_round = tf.round(tf.clip_by_value(255 * (self.cur_img), 0, 255))
                self.recon_image_round = tf.round(tf.clip_by_value(255 * (self.recon_image), 0, 255))
                self.psnr = tf.reduce_mean(tf.image.psnr(self.cur_img_round, self.recon_image_round, max_val=255))
                self.ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(self.cur_img_round, self.recon_image_round, max_val=255))
                self.ms_ssim_db = -10 * tf.log(tf.clip_by_value(1.0 - self.ms_ssim, 1E-10, 1.0)) / tf.log(10.0)
                
                if PConfig.loss_metric=="PSNR":
                    self.distortion_loss = self.mse_loss
                if PConfig.loss_metric=="MAE":
                    self.distortion_loss = self.mae_loss 
                if PConfig.loss_metric=="SSIM":
                    self.distortion_loss = 1-self.ms_ssim_d
                if PConfig.loss_metric=="L1MIX":
                    self.distortion_loss = 0.8*self.mae_loss+(1-0.8)/2*(1-self.ms_ssim_d)    
                if PConfig.loss_metric=="L2MIX":
                    self.distortion_loss = 0.976*self.mse_loss+0.024*(1-self.ms_ssim_d)

            # self.total_loss = self.lambda_train * self.distortion_loss + self.rate_loss
            self.total_loss = self.rate_loss
            self.cal_gradients()
            ########################以下用来给test或者PB提供简单的变量名##########################
            # self.feature_round = self.Net.feature_round
            # self.z_round = self.Net.z_round
            # self.recon_sigma = self.Net.recon_sigma
            # self.image_shape = self.Net.image_shape

            # self.identity_pb()
            ########################以上用来给PB提供简单的变量名##########################

       
    def summary(self):

        with tf.name_scope("summary") as scope:

            tf.summary.image("input_image", self.cur_img, 1)
            tf.summary.histogram("input_image", self.cur_img)            
            tf.summary.image("recon_image", self.clip_recon_image, 1)
            tf.summary.histogram("recon_image", self.clip_recon_image)
            tf.summary.scalar("psnr", self.psnr)
            tf.summary.scalar("mse_loss", self.mse_loss)
            tf.summary.scalar("total_loss", self.total_loss)

            self.summary_op = tf.summary.merge_all()

    
    def cal_gradients(self):

        with tf.name_scope("cal_grad"):
            var_all = tf.trainable_variables()
            ###################### 是否需要对所有参数都进行梯度更新 ########################
            if PConfig.opt_var_all:
                var_network = var_all
            else:
                print("***************** the train_ops is ****************************")
                var_network = []
                for i in var_all:
                    if PConfig.var_part_str in i.name:                        
                        if self.print_opt_var:
                            print(i.name)
                        var_network.append(i)
                self.print_opt_var = False # 在构建可训练变量中只有第一次GPU图构建的时候会打印变量名
                print("****************************************************************\n")
            grads = tf.gradients(self.total_loss, var_network)
            clipped_grads, _= tf.clip_by_global_norm(grads, 5)
            self.grad_and_vars = zip(clipped_grads, var_network)


    def create_session(self):
        # create the session and load parameters stage by stage
        config_device = tf.ConfigProto(allow_soft_placement=True)
        config_device.gpu_options.allow_growth = True
        if PConfig.use_frozen_model:
            self.sess = tf.Session(config=config_device)
            self.module_file = tf.train.latest_checkpoint(PConfig.checkpoint_dir)
        else:
            self.sess = tf.Session(config=config_device)

            print("\n\n****************************init****************************")
            self.sess.run(tf.global_variables_initializer())
            self.var_network = tf.trainable_variables()

            if PConfig.pretrain:
                self.load_pretrain()
            ################################# below is train or test load ##################################
            if self.is_train:
                self.saver = tf.train.Saver(max_to_keep=PConfig.max_ckpt_num)
                self.summary_writer = tf.summary.FileWriter(PConfig.checkpoint_dir + 'train', self.sess.graph)
            else:
                # test的时候只导入所有参数，而不导入训练的参数，如果有BN等，需要注意这样导不进去的
                
                var_1 = []
                for i in self.var_network:
                    if PConfig.var_part_str in i.name:                        
                        var_1.append(i)
                self.saver = tf.train.Saver(var_1, max_to_keep=PConfig.max_ckpt_num)
                # self.saver = tf.train.Saver(self.var_network, max_to_keep=PConfig.max_ckpt_num)

            self.module_file = tf.train.latest_checkpoint(PConfig.checkpoint_dir)
            print("the restored checkpoint is %s" % self.module_file)
            if self.module_file != None:
                self.saver.restore(self.sess, self.module_file)


    def load_pretrain(self):
        ###################### 第1次导入预训练好的实际模型 ########################
        print("***************** the pretrain mode is used ****************************")
        if PConfig.pre_var_all:
            var_pretrain = self.var_network
            print("***************** pretain all vars ****************************\n\n")
        else:
            var_pretrain = [] # 筛选特定变量导入
            for i in self.var_network:
                # print(i.name, "\n\n")
                if PConfig.pre_var_no_str not in i.name: # 名字里包含 pre_var_no_str 不导入预训练的模型 
                    print(i.name)
                    var_pretrain.append(i)
            print("***************** the aboves are the pretain partial vars ****************************\n\n")
        
        saver_pre = tf.train.Saver(var_pretrain)
        module_file = tf.train.latest_checkpoint(PConfig.checkpoint_dir_pre)
        print("the PConfig.pretrained checkpoint is %s" % module_file)
        if module_file != None:
            saver_pre.restore(self.sess, module_file)
        #######################################################################


    def import_graph(self):
        """PB模型不用于训练，只用于固化测试，这里提供将节点名转换为类通用名
        Outputs: 

        """

        # Load the frozen file and parse it
        with tf.gfile.GFile(PConfig.pb_name, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
            print('read from %s' % PConfig.pb_name)

        # import graph def
        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )
        # 类通用名
        self.pb_tensor()
    

    def summary_write(self, actual_step, feed_dict_sum):

        summary_merge = self.sess.run(self.summary_op, feed_dict=feed_dict_sum)
        self.summary_writer.add_summary(summary_merge, actual_step)


    def ckpt_write(self, actual_step):

        self.saver.save(self.sess, os.path.join(PConfig.checkpoint_dir, "model"), global_step=actual_step, write_meta_graph=False)


    def var_print(self):

        var_current = tf.trainable_variables() 
        var_num = 0
        with open(os.path.join(PConfig.info_dir, "/info.txt"), "a+") as f:
            f.write("the trainable variable is:\n\n")
            for i in range(len(var_current)):
                print(var_current[i].name,var_current[i].shape)
                f.write(var_current[i].name)
                f.write(str(var_current[i].shape))
                f.write("\n")
                var_num += np.prod(var_current[i].shape)
            print("the number of model variable:",var_num)
            print("******************************************************\n\n")
            f.write("the number of model variable:%s\n\n"%var_num)


    def identity_pb(self):

        tf.identity(self.feature_round, "feature_round")
        tf.identity(self.z_round, "z_round")
        tf.identity(self.clip_recon_image, "clip_recon_image")
        tf.identity(self.recon_sigma, "recon_sigma")

        tf.identity(self.mse_loss, "mse_loss")
        tf.identity(self.psnr, "psnr")
        tf.identity(self.total_loss, "total_loss")
        tf.identity(self.ms_ssim, "ms_ssim")
        tf.identity(self.bpp, "bpp")
        tf.identity(self.bpp_y, "bpp_y")


    def pb_tensor(self):

        self.input_image = self.graph.get_tensor_by_name("IModel/input_image:0")
        self.image_shape = self.graph.get_tensor_by_name("IModel/Shape:0")

        self.var_iclr18 = self.graph.get_tensor_by_name("IModel/var_iclr18:0")
        self.feature_round = self.graph.get_tensor_by_name("IModel/feature_round:0")
        self.z_round = self.graph.get_tensor_by_name("IModel/z_round:0")
        self.clip_recon_image = self.graph.get_tensor_by_name("IModel/clip_recon_image:0")        
        self.recon_sigma = self.graph.get_tensor_by_name("IModel/recon_sigma:0")

        self.mse_loss = self.graph.get_tensor_by_name("IModel/mse_loss:0")
        self.psnr = self.graph.get_tensor_by_name("IModel/psnr:0")
        self.total_loss = self.graph.get_tensor_by_name("IModel/psnr:0")        
        self.ms_ssim = self.graph.get_tensor_by_name("IModel/ms_ssim:0")
        self.bpp = self.graph.get_tensor_by_name("IModel/bpp:0")
        self.bpp_y = self.graph.get_tensor_by_name("IModel/bpp_y:0")
