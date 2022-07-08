# I_model
echo -e "##################################\nstart downloading the model"
mkdir -p I_model/CM_PSNR && cd I_model/CM_PSNR
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/I_model/CM_PSNR/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd - && echo -e "finish downloading the model\n##################################\n"

mkdir -p I_model/CM_SSIM && cd I_model/CM_SSIM
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/I_model/CM_SSIM/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd -

mkdir -p I_model/NoCM_PSNR && cd I_model/NoCM_PSNR
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/I_model/NoCM_PSNR/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd -

# P_model
mkdir -p P_model/NoSPM_PSNR && cd P_model/NoSPM_PSNR
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/P_model/NoSPM_PSNR/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd -

mkdir -p P_model/NoSPM_SSIM && cd P_model/NoSPM_SSIM
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/P_model/NoSPM_SSIM/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd -

mkdir -p P_model/STPM_PSNR && cd P_model/STPM_PSNR
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/P_model/STPM_PSNR/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd -

mkdir -p P_model/STPM_SSIM && cd P_model/STPM_SSIM
wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/IPCodec/P_model/STPM_SSIM/checkpoint.tar.gz
tar -xzvf checkpoint.tar.gz && rm -rf checkpoint.tar.gz
cd -