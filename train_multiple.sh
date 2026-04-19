#!/bin/bash
# train_multiple.sh
# 3-DF, 5-LA, 0-ITW/DFADD

# 定义通用变量
ALGO=0

# sequence1="No1_Stack-Mamba"
# sequence2="No2_Mix-Mamba-GDN"
# sequence3="No3_Mix-GDN-Mamba"
sequence4="No4_Mix-n_Mamba-1MLP" # MamBo-1
sequence5="No5_Mix-n_Mamba-1MHA" # MamBo-2
# sequence6="No6_Mix-n_Mamba-1MLP-1MHA"
# sequence7="No7_Mix-n_Mamba-1MHA-1MLP"
# sequence8="No8_Mix-1MHA-n_Mamba-1MLP"
sequence9="No9_Mix-n_Mamba-1MLP-n_Mamba-1MHA" # MamBo-4
# sequence10="No10_Mix-n_Mamba-1MHA-n_Mamba-1MLP"
# sequence11="No11_Mix-1MHA-n_Mamba-1MLP-n_Mamba"
sequence12="No12_Mix_n_Mamba_1MLP_1MHA_1MLP" # MamBo-3

# example
python train_test_score_AIO.py --type train --dataset asv19 --task xlsr_Mamba1 --block-cls Mamba1 --block-layers 5 --n-mamba 1 --block-sequence $sequence9 --algo 5 --save-checkpoint --out-fold ./outputs/ --save-model
python train_test_score_AIO.py --type test --dataset asv21la --task xlsr_Mamba1 --block-cls Mamba1 --block-layers 5 --n-mamba 1 --block-sequence $sequence9 --algo 5 --save-checkpoint --out-fold ./outputs/
python train_test_score_AIO.py --type score --dataset asv21la --task xlsr_Mamba1 --block-cls Mamba1 --block-layers 5 --n-mamba 1 --block-sequence $sequence9 --algo 5 --save-checkpoint --out-fold ./outputs/


# 加个提示
echo "All tasks launched!"