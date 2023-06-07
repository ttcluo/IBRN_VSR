#! /bin/bash

# --------------------------------------------------------------------------------------#
#                             测试批量化的执行多种代码                                  --#
# --------------------------------------------------------------------------------------#

data_name=$1

echo $data_name

if [ "vimeo90k" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_Vimeo90K.yml

elif [ "reds" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_REDS.yml

elif [ "real" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_REAL.yml

elif [ "vid4" = $data_name ]; then
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_Vid4.yml

else
    echo "Not found $data_name."
fi

echo "All $data_name has been run!"