#  python3 test_BRATS.py \
#  --dataroot ../DATASET/ResViT/BraSyn/fold1 \
#  --name BraSyn_unified_3_to_1_fold1_finetune \
#  --gpu_ids 0 \
#  --model test \
#  --which_model_netG resvit \
#  --dataset_mode uni \
#  --norm batch \
#  --phase test \
#  --output_nc 1 \
#  --input_nc 1 \
#  --fineSize 256 \
#  --loadSize 256 \
#  --results_dir results/ \
#  --checkpoints_dir checkpoints/ \
#  --which_epoch latest \
#  --no_flip \
#  --test_modality t1c \

#  python3 test_BRATS.py \
#  --dataroot ../DATASET/BRATS2023/2D_data_png/fold_0/ \
#  --name BraSyn_unified_3_to_1_fold0_pretrain_H100 \
#  --gpu_ids 0 \
#  --model test \
#  --which_model_netG res_cnn \
#  --dataset_mode uni \
#  --phase test \
#  --results_dir results/ \
#  --checkpoints_dir checkpoints/ \
#  --which_epoch latest \
#  --test_modality t1n \


python3 metric_BRATS.py \ 
python3 metric_BRATS_copy.py \ 
python3 metric_BRATS_copy3.py 