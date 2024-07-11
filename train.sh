################################
######### CHECK LIST ###########
################################
# name
# model
# which_model_netG
# pre_trained_path
# dataset_mode
################################

python3 train.py \
    --dataroot ../DATASET/ResViT/BraSyn/fold1 \
    --name BraSyn_unified_3_to_1_fold1_pretrain \
    --gpu_ids 0 \
    --model resvit_uni \
    --which_model_netG res_cnn \
    --which_direction AtoB \
    --lambda_A 100 \
    --dataset_mode uni \
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
    --lr 0.002 \
    --no_flip \
    --continue_train \
    --display_port 10100 \
    