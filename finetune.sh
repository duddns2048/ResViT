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
    --dataroot ../DATASET/ResViT/BraSyn/ \
    --name BraSyn_unified_t1c,t2w_t2f_finetune \
    --gpu_ids 0 \
    --model resvit_uni \
    --which_model_netG resvit \
    --which_direction AtoB \
    --lambda_A 100 \
    --dataset_mode uni \
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
    --pre_trained_path checkpoints/BraSyn_unified_t1c,t2w_t2f_finetune/latest_net_G.pth \
    --display_id 1 \
    --lr 0.002 \
    --no_flip \
    --continue_train \
    --which_epoch latest \
    --epoch_count 35 \
    --avail_fn 1110 \
    --display_port 10003 \
    