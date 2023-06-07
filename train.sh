#! /bin/bash

# --------------------------------------------------------------------------------------#
#                             测试批量化的执行多种代码                                  --#
# --------------------------------------------------------------------------------------#

data_name=$1

echo $data_name

if [ "vimeo90k" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_Vimeo90K.yml

elif [ "reds" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REDS.yml

elif [ "reds_e" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REDS_enhanced.yml

elif [ "reds_2" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REDS_enhanced_2.yml

elif [ "real" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REAL.yml

elif [ "real_e" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REAL_enhanced.yml

elif [ "ttvsr" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REDS_ttvsr.yml

elif [ "reds_swin" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REDS_swin.yml

else
    echo "Not found $data_name."
fi

echo "All $data_name has been run!"
