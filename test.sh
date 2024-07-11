 ####
 # args you must insert by typing not default
 # dataroot
 # name
 # model
 # which model netG
 # dataset_mode
 # phase
 # test_modality: t1n t1c t2w t2f
 ###

 python3 test_BRATS.py \
 --dataroot ../DATASET/ResViT/BraSyn/fold0 \
 --name  BraSyn_unified_3to1_finetune \
 --gpu_ids 0 \
 --model test \
 --which_model_netG resvit \
 --dataset_mode uni \
 --norm batch \
 --phase test \
 --fineSize 256 \
 --loadSize 256 \
 --checkpoints_dir checkpoints/ \
 --no_flip \
 --test_modality t1n

 python3 test_BRATS.py \
 --dataroot ../DATASET/ResViT/BraSyn/fold0 \
 --name  BraSyn_unified_3to1_finetune \
 --gpu_ids 0 \
 --model test \
 --which_model_netG resvit \
 --dataset_mode uni \
 --norm batch \
 --phase test \
 --fineSize 256 \
 --loadSize 256 \
 --checkpoints_dir checkpoints/ \
 --no_flip \
 --test_modality t2w

 python3 test_BRATS.py \
 --dataroot ../DATASET/ResViT/BraSyn/fold0 \
 --name  BraSyn_unified_3to1_finetune \
 --gpu_ids 0 \
 --model test \
 --which_model_netG resvit \
 --dataset_mode uni \
 --norm batch \
 --phase test \
 --fineSize 256 \
 --loadSize 256 \
 --checkpoints_dir checkpoints/ \
 --no_flip \
 --test_modality t2f

python3 metric_BRATS.py \