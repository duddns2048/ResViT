################################
######### CHECK LIST ###########
################################
# name
# model
# which_model_netG
# dataset_mode
# display_port
# input_nc
# G_only_L1
################################

# T2 -> Flair | pretrain | G_L1_only o
python3 train.py \
 --dataroot ../DATASET/ResViT/BRaTs/ \
 --name BraTS_T2_flair_pretrain_G_only_L1 \
 --gpu_ids 0 \
 --model resvit_one \
 --which_model_netG res_cnn \
 --which_direction AtoB \
 --lambda_A 100 \
 --dataset_mode unaligned \
 --norm batch \
 --pool_size 0 \
 --output_nc 1 \
 --input_nc 1 \
 --loadSize 256 \
 --fineSize 256 \
 --niter 50 \
 --niter_decay 50 \
 --save_epoch_freq 1 \
 --checkpoints_dir checkpoints/ \
 --display_id 1 \
 --lr 0.0002 \
 --display_port 10017 \
 --no_flip \
 --G_only_L1


# T2 -> Flair | finetune | G_L1_only o
python3 train.py \
    --dataroot ../DATASET/ResViT/BRaTs/ \
    --name BraTS_T2_flair_finetune_G_only_L1 \
    --gpu_ids 0 \
    --model resvit_one \
    --which_model_netG resvit \
    --which_direction AtoB \
    --lambda_A 100 \
    --dataset_mode unaligned \
    --norm batch \
    --pool_size 0 \
    --output_nc 1 \
    --input_nc 1 \
    --loadSize 256 \
    --fineSize 256 \
    --niter 25 \
    --niter_decay 25 \
    --save_epoch_freq 1 \
    --checkpoints_dir checkpoints/ \
    --display_id 1 \
    --pre_trained_transformer 1 \
    --pre_trained_resnet 1 \
    --pre_trained_path checkpoints/BraTS_T2_flair_pretrain/latest_net_G.pth \
    --lr 0.001 \
    --display_port 10018 \
    --no_flip \
    --G_only_L1
