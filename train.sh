################################
######### CHECK LIST ###########
################################
# name
# model
# which_model_netG
# pre_trained_path
# dataset_mode
################################

###template###

# python3 train.py \
#     \ "base options" \
#     --dataroot ../DATASET/BRATS2023/2D_data_png/fold_0 \
#     --name ResViT_unified \
#     --which_model_netG res_cnn \
#     --dataset_mode uni \
#     --model resvit_uni \
#     --batchSize 50 \
#     --display_port 10000 \
#     \
#     \ "train options" \
#     --niter 50 \
#     --niter_decay 50 \
#     --lr_policy lambda \
#     \
#     --validation_freq 1 \
#     --visdom_upload_freq 1 \
#     --print_freq 1 \
    
#     \# --continue_train \
#     \# --epoch_count \
#     \# --which_epoch \
    
#     \ "finetune" \
#     \# --pre_trained_path \
#     \# --pre_trained_resnet \
#     \# --lr \
#     \
#     --phase train \
#     --fold 0 \
#     --no_lsgan 0 \
#     \ # --randomsampler \
#     \ # --G_only_L1 \
#     --num_cases 25 

python3 train.py \
    --dataroot ../DATASET/BRATS2023/2D_data_png/fold_0 \
    --name ResViT_unified \
    --which_model_netG res_cnn \
    --dataset_mode uni \
    --model uni \
    --batchSize 20 \
    --display_port 10000 \
    \
    --niter 50 \
    --niter_decay 50 \
    --lr_policy lambda \
    \
    --validation_freq 1 \
    --visdom_upload_freq 1 \
    --print_freq 1 \
        \
    --fold 0 \
    --num_cases 25 

    







# python3 train.py \
#     --phase train \
#     --dataroot ../DATASET/BRATS2023/2D_data_png/fold_0 \
#     --fold 0 \
#     --name "BraSyn_unified_3_to_1_fold0_finetune_H10_case50" \
#     --model resvit_uni \
#     --which_model_netG resvit \
#     --dataset_mode uni \
#     --batchSize 50 \
#     --validation_freq 1 \
#     --print_freq 1 \
#     --lr 0.001 \
#     --niter 50 \
#     --which_epoch latest \
#     --pre_trained_resnet 1 \
#     --pre_trained_path ./checkpoints/BraSyn_unified_3_to_1_fold0_pretrain_H10_case50/latest_net_G.pth \
#     --checkpoints_dir checkpoints/ \
#     --display_port 10032 \
##



# python train.py \
#     --dataroot ../DATASET/BRATS2023/2D_data_png/fold_0\
#     --name BraSyn_unified_3_to_1_fold0_finetune_H100_prove\
#     --model uni\
#     --dataset_mode uni\
#     --which_model_netG resvit\
#     --batchSize 50\
#     \
#     --niter 25\
#     --niter_decay 25\
#     --lr_policy lambda\
#     \
#     --validation_freq 1\
#     --visdom_upload_freq 1\
#     --print_freq 1\
#     --phase train\
#     --fold 0\
#     --pre_trained_path ./checkpoints/BraSyn_unified_3_to_1_fold0_pretrain_H100_prove/latest_net_G.pth \
#     --pre_trained_resnet 1\
#     --lr 0.001\
#     --display_port 10704


# python train.py \
#     --dataroot ../DATASET/BRATS2023/2D_data_png/fold_0\
#     --name BraSyn_unified_3_to_1_fold0_pretrain_H100_prove\
#     --model uni\
#     --dataset_mode uni\
#     --which_model_netG resvit\
#     --batchSize 50\
#     \
#     --niter 25\
#     --niter_decay 25\
#     --lr_policy lambda\
#     \
#     --validation_freq 1\
#     --visdom_upload_freq 1\
#     --print_freq 1\
#     --phase train\
#     --fold 0\
#     --display_port 10703