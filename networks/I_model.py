# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import numpy as np
import tensorflow as tf

from I_config import IConfig
from networks.I_network import INet
from networks.modules.entropy.estimator import Entropy

from utils.train_pipeline import InputPipeline
from utils.train_dataloader import InputDataloader
from utils.dataloader import Dataloader
from utils.utils import average_gradients


class IModel(object):
    
    def __init__(self, is_train):

        self.is_train = is_train
        self.total_batch_size = IConfig.total_batch_size if is_train else 1
        self.Entr = Entropy()
        self.print_opt_var = True

    def build(self):

        with tf.Graph().as_default() as self.graph:
            
            # 使用固话模型的时候可以简单调用graph导入
            if IConfig.use_frozen_model:
                self.import_graph()
            elif self.is_train:
                self.build_dataset()
                self.build_train()
                self.summary()
                self.var_print()
            # 固化时单独固话解码端
            # elif IConfig.freeze_decoder:
            #     self.build_model_decoder()
            else:
                self.build_model()

            # create a session and load the checkpoint
            self.create_session() 
        
        return self.graph, self.sess


    def build_dataset(self):
        
        print("**************** the dataset information ****************")
        print("the train dir is %s" % IConfig.train_set_dir)
        # train_loader = InputDataloader(IConfig.train_set_dir, IConfig.height, IConfig.width, 16, self.total_batch_size, model_type="P") # dataloader的方式
        train_loader = InputPipeline(IConfig.train_set_dir, IConfig.height, IConfig.width, 64, self.total_batch_size, model_type="I") # pipeline的方式
        self.train_len = train_loader.file_len
        self.train_next = train_loader.train_image_batch

        print("the valid dir is %s" % IConfig.valid_set_dir)
        valid_loader = Dataloader(IConfig.valid_set_dir, 8, self.total_batch_size, model_type="valid") # pipeline的方式
        self.valid_len = valid_loader.file_len
        self.init_valid = valid_loader.initializer
        self.valid_next = valid_loader.test_image_batch


    def build_train(self):
        
        self.global_step = tf.train.get_or_create_global_step()
        self.initial_learning_rate = tf.train.piecewise_constant(self.global_step, \
            boundaries=IConfig.boundaries, values=IConfig.multi_learning_rates)
        self.LR = self.initial_learning_rate
        opt = tf.train.AdamOptimizer(self.initial_learning_rate)
        
        if IConfig.gpus_list:
            models = []
            for index, gpu_id in enumerate(IConfig.gpus_list):
                with tf.device('/gpu:%d' % index):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        print('Building Graph on  %dth GPU, id %d' % (index+1, gpu_id))
                        self.build_model()
                        if IConfig.is_multi:
                            models.append((self.input_image_in, self.lambda_onehot, self.grad_and_vars))
                        else:
                            models.append((self.input_image_in, self.grad_and_vars))
            
            if IConfig.is_multi:
                self.tower_input, self.tower_onehot, tower_grads = zip(*models)
            else:
                self.tower_input, tower_grads = zip(*models)

            self.train_ops = opt.apply_gradients(average_gradients(tower_grads), global_step=self.global_step)

        else:
            with tf.variable_scope(tf.get_variable_scope()):
                print('Graph was built on CPU %d')
                self.build_model()
                self.train_ops = opt.apply_gradients(self.grad_and_vars, global_step=self.global_step)


    def build_model(self):

        with tf.variable_scope('IModel'):

            if not IConfig.freeze_fix_img:
                self.input_image_in = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name="input_image")
            else:
                self.input_image_in = tf.placeholder(dtype=tf.uint8, shape=IConfig.fix_size, name="input_image")
                
            if IConfig.is_multi:
                self.lambda_onehot = tf.placeholder(dtype=tf.float32, shape=[len(IConfig.lambda_list)], name="lambda_onehot")
                self.lambda_train = tf.reduce_sum(tf.multiply(self.lambda_onehot, IConfig.lambda_list))
            else:
                self.lambda_train = tf.cast(tf.convert_to_tensor(IConfig.train_lambda), dtype=tf.float32)
                   
            self.input_image = tf.cast(self.input_image_in, dtype=tf.float32)
            self.input_image = tf.div(self.input_image, 255, name="normalization")
            
            self.INet = INet(is_train=self.is_train)
            if IConfig.is_multi:
                self.recon_image = self.INet(self.input_image, self.lambda_onehot)
            else:
                self.recon_image = self.INet(self.input_image)
            
                
            self.clip_recon_image = tf.clip_by_value(self.recon_image, 0, 1)

            with tf.name_scope("rate"):
                total_bits_y = self.Entr.entropy_y(self.INet.y_hat, self.INet.recon_sigma)
                total_bits_z = self.Entr.entropy_z(self.INet.z_hat)

                im_shape = tf.cast(tf.shape(self.input_image), tf.float32)
                self.bpp_y = total_bits_y / (im_shape[0]  * im_shape[1] * im_shape[2])
                # print(type(total_bits_y), type(self.total_batch_size), type(im_shape[1]), type(im_shape[2]))
                self.bpp_z = total_bits_z / (im_shape[0]  * im_shape[1] * im_shape[2])
                self.bpp = self.bpp_y + self.bpp_z #the estimatied bit rates
                self.rate_loss = self.bpp
            
            with tf.name_scope("distortion"):
                self.mse_loss = tf.reduce_mean(tf.square((self.input_image) - (self.recon_image)))
                self.mae_loss = tf.reduce_mean(tf.abs((self.input_image) - (self.recon_image)))
                self.ms_ssim_d = tf.reduce_mean(tf.image.ssim_multiscale(self.input_image, self.recon_image, max_val=1.0))

                self.input_image_round = tf.round(tf.clip_by_value(255 * (self.input_image), 0, 255))
                self.recon_image_round = tf.round(tf.clip_by_value(255 * (self.recon_image), 0, 255))
                self.psnr = tf.reduce_mean(tf.image.psnr(self.input_image_round, self.recon_image_round, max_val=255))
                self.ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(self.input_image_round, self.recon_image_round, max_val=255))
                self.ms_ssim_db = -10 * tf.log(tf.clip_by_value(1.0 - self.ms_ssim, 1E-10, 1.0)) / tf.log(10.0)
                
                if IConfig.loss_metric=="PSNR":
                    self.distortion_loss = self.mse_loss
                if IConfig.loss_metric=="MAE":
                    self.distortion_loss = self.mae_loss 
                if IConfig.loss_metric=="SSIM":
                    self.distortion_loss = 1-self.ms_ssim_d
                if IConfig.loss_metric=="L1MIX":
                    self.distortion_loss = 0.8*self.mae_loss+(1-0.8)/2*(1-self.ms_ssim_d)    
                if IConfig.loss_metric=="L2MIX":
                    self.distortion_loss = 0.976*self.mse_loss+0.024*(1-self.ms_ssim_d)

            # self.total_loss = self.lambda_train * self.distortion_loss + self.rate_loss
            self.total_loss = self.distortion_loss
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

            tf.summary.image("input_image", self.input_image, 1)
            tf.summary.histogram("input_image", self.input_image)            
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
            if IConfig.opt_var_all:
                var_network = var_all
            else:
                print("***************** the train_ops is ****************************")
                var_network = []
                for i in var_all:
                    if IConfig.var_part_str in i.name:                        
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
        if IConfig.use_frozen_model:
            self.sess = tf.Session(config=config_device)
            self.module_file = tf.train.latest_checkpoint(IConfig.checkpoint_dir)
        else:
            self.sess = tf.Session(config=config_device)

            print("\n\n****************************init****************************")
            self.sess.run(tf.global_variables_initializer())
            self.var_network = tf.trainable_variables()

            if IConfig.pretrain:
                self.load_pretrain()
            ################################# below is train or test load ##################################
            if self.is_train:
                self.saver = tf.train.Saver(max_to_keep=IConfig.max_ckpt_num)
                self.summary_writer = tf.summary.FileWriter(IConfig.checkpoint_dir + 'train', self.sess.graph)
            else:
                # test的时候只导入所有参数，而不导入训练的参数，如果有BN等，需要注意这样导不进去的
                self.saver = tf.train.Saver(self.var_network, max_to_keep=IConfig.max_ckpt_num)

            self.module_file = tf.train.latest_checkpoint(IConfig.checkpoint_dir)
            print("the restored checkpoint is %s" % self.module_file)
            if self.module_file != None:
                self.saver.restore(self.sess, self.module_file)


    def load_pretrain(self):
        ###################### 第1次导入预训练好的实际模型 ########################
        print("***************** the pretrain mode is used ****************************")
        if IConfig.pre_var_all:
            var_pretrain = self.var_network
            print("***************** pretain all vars ****************************\n\n")
        else:
            var_pretrain = [] # 筛选特定变量导入
            for i in self.var_network:
                # print(i.name, "\n\n")
                if IConfig.pre_var_no_str not in i.name:
                    print(i.name)
                    var_pretrain.append(i)
            print("***************** the aboves are the pretain partial vars ****************************\n\n")
        
        saver_pre = tf.train.Saver(var_pretrain)
        module_file = tf.train.latest_checkpoint(IConfig.checkpoint_dir_pre)
        print("the IConfig.pretrained checkpoint is %s" % module_file)
        if module_file != None:
            saver_pre.restore(self.sess, module_file)
        #######################################################################


    def import_graph(self):
        """PB模型不用于训练，只用于固化测试，这里提供将节点名转换为类通用名
        Outputs: 

        """

        # Load the frozen file and parse it
        with tf.gfile.GFile(IConfig.pb_name, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
            print('read from %s' % IConfig.pb_name)

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

        self.saver.save(self.sess, os.path.join(IConfig.checkpoint_dir, "model"), global_step=actual_step, write_meta_graph=False)


    def var_print(self):

        var_current = tf.trainable_variables() 
        var_num = 0
        with open(os.path.join(IConfig.info_dir, "/info.txt"), "a+") as f:
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
