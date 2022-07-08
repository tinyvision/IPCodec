# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import numpy as np
import tensorflow as tf

class ModelIConfig(object):
    

    def __init__(self):

        # whether use pre-trained models
        self.pretrain = 1  # 1: use；0：not use
        self.pre_var_all = 1 # 1：load all parameters；0：load parts of parameters。
        self.pre_var_no_str = "mask_w" # exclude the parameters in the pre-trained model that are not loaded to the current model
        self.checkpoint_dir_pre = "./checkpoint/I_model/CM_PSNR"

        # whether to training all parameters 
        self.opt_var_all = 1 # 1: train all parameters；0：train parts of parameters
        self.var_part_str = "post_unet" # the name with the str will be trained 

        # netwrok parameters and training settings
        self.total_batch_size = 32 # batchsize
        self.boundaries = [1600000, 2100000, 2300000, 2400000] # boundaries of stagewise LRs
        self.multi_learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6] # stagewise LRs
        self.train_lambda = 5500 # single rate train lambda
        self.loss_metric = "PSNR" # metric_list = ["PSNR", "MAE", "SSIM","L1MIX","L2MIX"]
        self.is_post = True # add the Unet post network
        self.with_context_model = True # add the context model network
        self.is_multi = True # enable variable rate control
        self.alpha = [[0., 0.5, 1.], [1, 1, 1]]
        self.ch_E = [[256, 256, 256, 320], [256, 256, 256]] # the channel numbers of encoder and hyper encoder
        self.ch_D = [[256, 256, 640], [256, 256, 256, 3]] # the channel numbers of hyper decoder and decoder
        self.gpus_list = [4, 5, 6, 7] # the GPUIDs of train，must be divided by batchszie
        self.gpus_test =[0] # the GPUID of valid or test.
        self.show_step = 500 # iteration interval to show the training result
        self.max_ckpt_num = 20

        ###### imagesize ######
        self.height = 256
        self.width = 256

        ###### train and valid dataset ###### 
        self.train_set_dir = "/home/your_path/Datasets/Images/train"
        self.valid_set_dir = "/home/your_path/Datasets/Images/valid"

        ###### img test dataset ###### 
        self.test_set_dir = "./dataset/kodak/"

        ###### freeze ckpt to pb ###### 
        self.use_frozen_model = 0 # if use frozen pb model, set to 1
        self.freeze_fix_img = False # Whether use the fix image size to freeze the model
        self.fix_type={"720P":[1, 720, 1280, 3], "1080P":[1, 1080, 1920, 3]}
        self.fix_size = [1, 1920, 1080, 3] # fixed image size
        # pb name to freeze
        self.pb_name_freeze = "model_freeze.pb" # use freeze mode
        self.pb_name_constant = "model_constants.pb" # use constant mode
        self.pb_name_rt = "model_RT.pb" # use TensorRT mode
        self.freeze_decoder = 0 # Whether to freeze the decoder


    def cckpt(self, args): # modify the path and info
        if hasattr(args, "loss_metric"): self.loss_metric = args.loss_metric
        if hasattr(args, "is_post"): self.is_post = args.is_post
        if hasattr(args, "with_context_model"): self.with_context_model = args.with_context_model
        if hasattr(args, "is_multi"): self.is_multi = args.is_multi

        self.model_name = "%s_%s"%(args.model_name, self.loss_metric)
        if self.loss_metric == "PSNR":
            self.lambda_list = np.array([50, 160, 300, 480, 710, 1000, 1350, 1780, 2302, 2915], dtype=np.float32)
        if self.loss_metric == "SSIM":
            self.lambda_list = np.array([3, 8, 14, 20, 35, 52, 78, 98, 120, 145], dtype=np.float32)
        self.lambda_onehot = np.identity(len(self.lambda_list))

        self.ckpt_base = "./checkpoint/I_model" if args.work_dir is None else args.work_dir
        self.checkpoint_dir = os.path.join(self.ckpt_base, self.model_name)
        self.info_dir = os.path.join(self.ckpt_base, self.model_name, "info")
        self.rescon_dir = os.path.join(self.ckpt_base, self.model_name, "date/recon")
        self.bin_dir = os.path.join(self.ckpt_base, self.model_name, "date/bin")
        self.datatext_dir = os.path.join(self.info_dir, "test_kodak.txt")
        self.pb_path = os.path.join(self.ckpt_base, self.model_name, "pb")
        self.pb_name = self.pb_path + self.pb_name_freeze
        self.test_comment = "just for test"


IConfig = ModelIConfig()

