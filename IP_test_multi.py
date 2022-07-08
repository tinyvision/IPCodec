# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import glob
import argparse
import numpy as np
from time import time
from PIL import Image
import tensorflow as tf
from datetime import datetime

from IP_config import PConfig
from networks.IP_model import PModel
from utils.dataloader import Dataloader
from utils.utils import load_single_image


def main():
    #built the model
    PMod=PModel(is_train=False)
    graph, sess = PMod.build()

    if not os.path.isdir(PConfig.info_dir):
        os.makedirs(PConfig.info_dir)

    print("imgaeGenerator--%d"%IPPar.gpu)
    
    with graph.as_default():
        
        print("the test seq_name is %s" % seq_name)
        file = open(PConfig.datatext_dir, 'a+')
        file.write("\n#######################################################\n")
        file.write("Below is %s sequence test results (1I%dP)\n"%(seq_name, PConfig.test_GOP-1))
        file.write("#######################################################\n")
        file.close()
        ori_seq_dir = os.path.join(PConfig.test_set_dir, seq_name)
        print(ori_seq_dir)
        test_loader = Dataloader(ori_seq_dir, 8, batch_size, model_type="test") # pipeline的方式
        test_len = test_loader.file_len if PConfig.frames==9999 else PConfig.frames
        test_filenames = test_loader.test_filenames
        init_test = test_loader.initializer
        test_next = test_loader.test_image_batch

        index_random = PConfig.idx_test
        alpha_rand = PConfig.alpha_test 
        # alpha_rand = 0.7
        if PConfig.is_multi:
            l_onehot = alpha_rand*PConfig.lambda_onehot[index_random] + (1-alpha_rand)*PConfig.lambda_onehot[index_random+1]
            lambda_test = alpha_rand*PConfig.lambda_list[index_random] + (1-alpha_rand)*PConfig.lambda_list[index_random+1]
            PConfig.test_comment = "lambda_%d"%(lambda_test)
            print("\n\n =====> %s ======================" % PConfig.test_comment )
        else:
            lambda_test = PConfig.train_lambda
            PConfig.test_comment = "lambda_%d"%(lambda_test)

        sess.run(init_test) #  每次都重新开始初始化test iteration迭代器，重头开始

        average_psnr = 0.
        average_bpp = 0.
        average_estbpp = 0.
        average_estbpp_res = 0.
        average_estbpp_min = 0.
        average_msssim = 0.

        os.makedirs(PConfig.bin_dir, exist_ok=True)
        os.makedirs(os.path.join(PConfig.rescon_dir, seq_name), exist_ok=True)

        ####################################################################### 
        for i in range(test_len):

            tic = time()
            if i%PConfig.test_GOP == 0: # I帧直接读取已经完成的编解码信息
                frame_flag = "I"
                cur_img_batch = sess.run(test_next)
                image_name = os.path.basename(test_filenames[i])

                input_image_batch = np.concatenate((cur_img_batch, cur_img_batch, cur_img_batch), axis=2)
                if PConfig.is_multi:
                    feed_dict={PMod.input_image_in:input_image_batch, PMod.lambda_onehot:l_onehot}
                else:
                    feed_dict={PMod.input_image_in:input_image_batch}
                psnr, ms_ssim, bpp, bpp_y = sess.run([PMod.psnr, PMod.ms_ssim, PMod.bpp, PMod.bpp_y], feed_dict=feed_dict)
                bpp_res = bpp
                bpp_min = np.minimum(bpp, bpp_res)
            else:
                frame_flag = "P"
                cur_img_batch = sess.run(test_next)
                # pre_img_recon_batch = clipped_recon_image
                image_shape = np.shape(cur_img_batch)
                image_name = os.path.basename(test_filenames[i])
                image_recon_path = os.path.join(PConfig.rescon_dir, seq_name, image_name)

                input_image_batch = np.concatenate((pre_img_batch, pre_img_batch, cur_img_batch), axis=2)
                if PConfig.is_multi:
                    feed_dict={PMod.input_image_in:input_image_batch, PMod.lambda_onehot:l_onehot}
                else:
                    feed_dict={PMod.input_image_in:input_image_batch}

                psnr, ms_ssim, bpp, bpp_res, bpp_y_res, recon_image = sess.run([PMod.psnr, PMod.ms_ssim, PMod.bpp, PMod.bpp_res, PMod.bpp_y_res, PMod.clip_recon_image], feed_dict=feed_dict)
                tic1 = time()
                bpp_min = np.minimum(bpp, bpp_res)
                # print(bpp_res, bpp_y_res, bpp_res-bpp_y_res)
                #########################调用最新的编码文件###################################################
                # bin_path = PConfig.bin_dir + image_name.replace(".png",".bin")
                actual_total_bits = 0 # 这里还没有开始写实际编解码函数-entropy_encoding
                actual_bpp = actual_total_bits / (batch_size * image_shape[1] * image_shape[2])
                tic2 = time()

                # 将训练后的图像保存到data-recon-mini512中
                # print('The shape is Recon', clipped_recon_image.shape)
                # clipped_recon_image = (np.round(recon_image*255)).astype(np.uint8)
                # Image.fromarray(clipped_recon_image[0]).save(image_recon_path)
                # nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%2S')
                # print("%s %s : the nn time is %.2f s, the entropy time is %.2f s "%(nowtime, image_name, tic1-tic, tic2-tic1), image_shape)

            pre_img_batch = cur_img_batch
            
            average_psnr += psnr
            # average_bpp += actual_bpp
            average_estbpp += bpp
            average_estbpp_res += bpp_res
            average_estbpp_min += bpp_min
            average_msssim += ms_ssim

            print("%dth frame (%s) in %20s result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)"%(i, frame_flag, seq_name, bpp, bpp_res, bpp_min, psnr, ms_ssim))
            # txt_write[image_name] = [psnr_val, ms_ssim_np, 0, bpp_val]
            if i < 108:
                file = open(PConfig.datatext_dir, 'a+')
                file.write("%dth frame (%s) result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)\n"%(i, frame_flag, bpp, bpp_res, bpp_min, psnr, ms_ssim))
                file.close()

        print("lambda:%d, %s sequence test result, PSNR:%.3f, msssim:%.4f, Ibpp:%.4f, Resbpp:%.4f, Minbpp:%.4f\n\n" % (
            lambda_test, seq_name, 
            average_psnr / test_len,
            average_msssim / test_len,
            average_estbpp / test_len,
            average_estbpp_res / test_len,
            average_estbpp_min / test_len))
        file = open(PConfig.datatext_dir, 'a+')
        file.write("the number of PMod is %s %s\n" % (PMod.module_file, PConfig.test_comment))
        file.write("%20s sequence test result is PSNR:%.6f, msssim:%.6f, Ibpp:%.6f, Resbpp:%.6f, Minbpp:%.6f\n\n" % \
                (seq_name,
                average_psnr / test_len,
                average_msssim / test_len,
                average_estbpp / test_len,
                average_estbpp_res / test_len,
                average_estbpp_min / test_len))
        file.close()


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IP_test")

    # 节点及GPU使用
    parser.add_argument("--data", type=str, default="UVG", choices=["UVG", "HB", "HC", "HD", "HE", "MCL"],
                        help="test different data") 
    parser.add_argument("--seq_name", type=str, default="Beauty",
                        help="test different data") 
    parser.add_argument("--test_set_dir", type=str, default=PConfig.test_set_dir,
                        help="test different data")                                                 
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu") 
    parser.add_argument("--gop", type=int, default=108, choices=[2, 5, 6, 10, 12, 108],
                        help="test gop in video")    # UVG=12/100, HEVC=10/100
    parser.add_argument("--frames", type=int, default=9999, choices=[100, 108, 9999],
                        help="numbers of test frames in video") # UVG=9999, HEVC=100
    parser.add_argument("--idx_test", type=int, default=0, choices=range(10),
                        help="idx")    
    parser.add_argument("--alpha_test", type=float, default=0.5,
                        help="alpha range 0-1")
                        
    IPPar = parser.parse_args()
    #model and test config
    os.environ['CUDA_VISIBLE_DEVICES']='%d'%(IPPar.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batch_size = 1
    if IPPar.data == "UVG":
        PConfig.fix_size =[1, 1080, 1920*3, 3]
    elif IPPar.data == "HB":
        PConfig.fix_size =[1, 1080, 1920*3, 3]
    elif IPPar.data == "HC":
        PConfig.fix_size =[1, 480, 832*3, 3]
    elif IPPar.data == "HD":
        PConfig.fix_size =[1, 240, 416*3, 3]
    elif IPPar.data == "HE":
        PConfig.fix_size =[1, 720, 1280*3, 3]
    elif IPPar.data == "MCL":
        PConfig.fix_size =[1, 1080, 1920*3, 3]

    PConfig.is_post = True
    PConfig.test_set_dir = IPPar.test_set_dir
    PConfig.test_GOP = IPPar.gop
    PConfig.idx_test = IPPar.idx_test
    PConfig.alpha_test = IPPar.alpha_test
    PConfig.frames = IPPar.frames
    seq_name = IPPar.seq_name

    PConfig.datatext_dir = os.path.join(PConfig.info_dir, IPPar.data, \
                            "test_GOP%d_%s.txt"%(PConfig.test_GOP, seq_name))
    os.makedirs(os.path.dirname(PConfig.datatext_dir), exist_ok=True)

    main()
