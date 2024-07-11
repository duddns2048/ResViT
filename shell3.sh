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

# T2 -> Flair | train_all | G_L1_only x
python3 train.py \
 --dataroot ../DATASET/ResViT/BRaTs/ \
 --name BraTS_T2_flair_train_all \
 --gpu_ids 0 \
 --model resvit_one \
 --which_model_netG resvit \
 --which_direction AtoB \
 --lambda_A 100\
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
 --pre_trained_transformer 1 \
 --lr 0.0002 \
 --no_flip \
 --display_port 10019 \
#  --G_only_L1




# T1, T2 -> Flair | train_all | G_L1_only x
python3 train.py \
 --dataroot ../DATASET/ResViT/BRaTs/ \
 --name BraTS_T1_T2_flair_train_all \
 --gpu_ids 0 \
 --model resvit_many \
 --which_model_netG resvit \
 --which_direction AtoB \
 --lambda_A 100\
 --dataset_mode many \
 --norm batch \
 --pool_size 0 \
 --output_nc 1 \
 --input_nc 3 \
 --loadSize 256 \
 --fineSize 256 \
 --niter 50 \
 --niter_decay 50 \
 --save_epoch_freq 1 \
 --checkpoints_dir checkpoints/ \
 --display_id 1 \
 --pre_trained_transformer 1 \
 --lr 0.0002 \
 --no_flip \
 --display_port 10021 \
#  --G_only_L1

# T1, T2 -> Flair | train_all | G_L1_only o
python3 train.py \
 --dataroot ../DATASET/ResViT/BRaTs/ \
 --name BraTS_T1_T2_flair_train_all_G_only_L1 \
 --gpu_ids 0 \
 --model resvit_many \
 --which_model_netG resvit \
 --which_direction AtoB \
 --lambda_A 100\
 --dataset_mode many \
 --norm batch \
 --pool_size 0 \
 --output_nc 1 \
 --input_nc 3 \
 --loadSize 256 \
 --fineSize 256 \
 --niter 50 \
 --niter_decay 50 \
 --save_epoch_freq 1 \
 --checkpoints_dir checkpoints/ \
 --display_id 1 \
 --pre_trained_transformer 1 \
 --lr 0.0002 \
 --no_flip \
 --display_port 10022 \
 --G_only_L1