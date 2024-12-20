#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

root_dir="/root/dataset/nerf_example_data/nerf_synthetic"
# list="chair drums ficus hotdog lego materials mic ship"
list="lego"

for i in $list; do

    # 训练3DGS
    nohup \
        python train.py --eval \
        -s $root_dir/$i \
        -m output/NeRF_Syn/$i/3dgs \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        --lambda_mask_entropy 0.1 \
        --save_training_vis \
        --lambda_depth_var 1e-2 \
        >output/NeRF_Syn/$i/3dgs/log.txt 2>&1 &

    # 新颖视角合成
    nohup \
        python eval_nvs.py --eval \
        -m output/NeRF_Syn/${i}/3dgs \
        -c output/NeRF_Syn/${i}/3dgs/chkpnt30000.pth \
        >output/NeRF_Syn/$i/3dgs/log.txt 2>&1 &

    nohup \
        python train.py --eval \
        -s $root_dir/$i \
        -m output/NeRF_Syn/$i/neilf \
        -c output/NeRF_Syn/$i/3dgs/chkpnt30000.pth \
        --save_training_vis \
        --position_lr_init 0.000016 \
        --position_lr_final 0.00000016 \
        --normal_lr 0.001 \
        --sh_lr 0.00025 \
        --opacity_lr 0.005 \
        --scaling_lr 0.0005 \
        --rotation_lr 0.0001 \
        --iterations 40000 \
        --lambda_base_color_smooth 0 \
        --lambda_roughness_smooth 0 \
        --lambda_light_smooth 0 \
        --lambda_light 0.01 \
        -t neilf --sample_num 40 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01 \
        >output/NeRF_Syn/$i/neilf/log.txt 2>&1 &

    nohup \
        python eval_nvs.py --eval \
        -m output/NeRF_Syn/${i}/neilf \
        -c output/NeRF_Syn/${i}/neilf/chkpnt40000.pth \
        -t neilf \
        >output/NeRF_Syn/$i/neilf/log.txt 2>&1 &

done
