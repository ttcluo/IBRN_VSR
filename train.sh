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

elif [ "real" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_REAL.yml

else
    echo "Not found $data_name."
fi

echo "All $data_name has been run!"
