from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
import torch
import numpy as np
import SimpleITK as sitk 
import os
import json
import multiprocessing

# Define evaluation Metrics
psnr = PeakSignalNoiseRatio(data_range=1.0).cuda() #because we normalize to 0-1
ssim = StructuralSimilarityIndexMeasure().cuda()
mse = MeanSquaredError().cuda()

def is_case_id_in_file(case_id, file_path):
    with open(file_path, 'r') as f:
        file_contents = f.read()
    return case_id in file_contents


def __percentile_clip(input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True):
    """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
    Percentiles for normalization can come from another tensor.

    Args:
        input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
            If reference_tensor is None, the percentiles from this tensor will be used.
        reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
        p_min (float, optional): Lower end percentile. Defaults to 0.5.
        p_max (float, optional): Upper end percentile. Defaults to 99.5.
        strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

    Returns:
        torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
    """
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile

    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]

    return output_tensor

            

def compute_metrics(gt_image: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor, normalize=True):
    """Computes MSE, PSNR and SSIM between two images only in the masked region.

    Normalizes the two images to [0;1] based on the gt_image 0.5 and 99.5 percentile in the non-masked region.
    Requires input to have shape (1,1, X,Y,Z), meaning only one sample and one channel.
    For SSIM, we first separate the input volume to be tumor region and non-tumor region, then we apply regular SSIM on the complete volume. In the end we take
    the two volumes.

    Args:
        gt_image (torch.Tensor): The ground truth image (***.nii.gz)
        prediction (torch.Tensor): The inferred/predicted image
        mask (torch.Tensor): The segmentation mask (seg.nii.gz)
        normalize (bool): Normalizes the input by dividing trough the maximal value of the gt_image in the masked
            region. Defaults to True

    Raises:
        UserWarning: If you dimensions do not match the (torchmetrics) requirements: 1,1,X,Y,Z

    Returns:
        float: (SSIM_tumor, SSIM_non_tumor)
    """

    if not (prediction.shape[0] == 1 and prediction.shape[1] == 1):
        raise UserWarning(f"All inputs have to be 5D with the first two dimensions being 1. Your prediction dimension: {prediction.shape}")
    
    # Normalize to [0;1] individually after intensity clipping
    if normalize:
        gt_image = __percentile_clip(gt_image, p_min=0.5, p_max=99.5, strictlyPositive=True)
        prediction = __percentile_clip(prediction, p_min=0.5, p_max=99.5, strictlyPositive=True)

    mask[mask>0] = 1
    mask = mask.type(torch.int64)
    
    gt_image = gt_image.cuda()
    prediction = prediction.cuda()
    mask = mask.cuda()
    
    # PSNR - apply on full image
    PSNR_full = psnr(preds=prediction, target=gt_image)
    SSIM_full = ssim(preds=prediction, target=gt_image)

    # Get Infill region (we really are only interested in the infill region)
    prediction_tumor = prediction * mask
    gt_image_tumor = gt_image * mask

    prediction_non_tumor = prediction * (1-mask)
    gt_image_non_tumor = gt_image * (1-mask)


    # PSNR - apply on complete masked image but only take values from masked region
    PSNR_tumor = psnr(preds=prediction_tumor, target=gt_image_tumor)
    PSNR_non_tumor = psnr(preds=prediction_non_tumor, target=gt_image_non_tumor)


    # SSIM - apply on complete masked image but only take values from masked region
    SSIM_tumor = ssim(preds=prediction_tumor, target=gt_image_tumor)
    SSIM_non_tumor = ssim(preds=prediction_non_tumor, target=gt_image_non_tumor)

    return float(PSNR_full), float(PSNR_tumor), float(PSNR_non_tumor), float(SSIM_full), float(SSIM_tumor), float(SSIM_non_tumor)




# test configuration
# fold_num = 0
# missing_modality = "t1c" # "t1c", "t1n", "t2f", "t2w"
def process_fold_modality(args):
    fold_num, missing_modality = args
    data_dir = '../DATASET/'

    with open(os.path.join(data_dir, 'splits_brats.json'), 'r') as f:
        folds_info = json.load(f)

    case_id_list = []
    mse_full_list = []
    mse_tumor_list = []
    mse_non_tumor_list = []
    psnr_full_list = []
    psnr_tumor_list = []
    psnr_non_tumor_list = []
    ssim_full_list = []
    ssim_tumor_list = []
    ssim_non_tumor_list = []

    eval_metric_list = [ 'psnr_full', 'psnr_tumor', 'psnr_non_tumor', 'ssim_full', 'ssim_tumor', 'ssim_non_tumor']
    
    metric_txt = f'./checkpoints/{exp_name}/results/{missing_modality}_metrics.txt'
    if os.path.exists(metric_txt):
        raise FileExistsError(f"파일 {metric_txt}가 이미 존재합니다.")
    
    if not os.path.isfile(metric_txt): # if file doesn't exist
        f = open(os.path.join(metric_txt), "a")
        f.write(f'case_ids')
        for dct in eval_metric_list:
            f.write(f'\t{dct}')
        f.write(f'\n')
        f.close()

    test_cases = folds_info[f'fold_{fold_num}']['test']
    image_folder = os.path.join(data_dir, 'BRATS2023/3D_data_gt')

    for test_case in test_cases:
        case_id_list.append(test_case)
        
        # # # if case id is already in metrics.txt file, then skip the loop
        # if is_case_id_in_file(test_case, f'/backup/BraSyn_tutorial/results/init_cv/test_latest/{fold_num}/{missing_modality}/metrics.txt'):
        #     print(f'Case id {test_case} is already in the file.')
        #     # read line and get metrics
        #     f = open(os.path.join(f'/backup/BraSyn_tutorial/results/init_cv/test_latest/{fold_num}/{missing_modality}/metrics.txt'), "r")
            
        #     continue

        
        case_path = os.path.join(image_folder,f'fold_{fold_num}','test', test_case)
        results_dir = f'./checkpoints/{exp_name}/results/fold{fold_num}/'
        results_path = os.path.join(results_dir, test_case)
        
        real_path = os.path.join(case_path, f'{test_case}-{missing_modality}.nii.gz')
        syn_path = os.path.join(results_path, f'{test_case}-{missing_modality}.nii.gz')
        seg_path = os.path.join(case_path, f'{test_case}-seg.nii.gz')

        print(syn_path)
        array_real = sitk.GetArrayFromImage(sitk.ReadImage(real_path))
        array_syn = sitk.GetArrayFromImage(sitk.ReadImage(syn_path))
        array_seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        
        # MSE calculation without extra package
        # array_real_norm = __percentile_clip(array_real, p_min=0.5, p_max=99.5, strictlyPositive=True)
        
        x_min = np.amin(array_real)
        x_max = np.amax(array_real)
        array_real_norm = (array_real - x_min) / (x_max - x_min)
        
        # MSE Full
        MSE_full = np.square(np.subtract(array_real_norm, array_syn)).mean() 
        
        # non-tumor region
        mask = array_seg.copy() 
        mask[mask>0] = 1
        mask = mask.astype(np.int64)

        MSE_healthy = np.square(np.subtract(array_real_norm * (1-mask), array_syn * (1-mask))).mean()
        MSE_tumor = np.square(np.subtract(array_real_norm * mask, array_syn * mask)).mean()       
        
        # 
        mse_full_list.append(MSE_full)
        mse_non_tumor_list.append(MSE_healthy)
        mse_tumor_list.append(MSE_tumor)

        array_real = array_real[np.newaxis, np.newaxis, ...]
        array_syn = array_syn[np.newaxis, np.newaxis, ...]
        array_seg = array_seg[np.newaxis, np.newaxis, ...]

        array_real = torch.from_numpy(array_real)
        array_syn = torch.from_numpy(array_syn)
        array_seg = torch.from_numpy(array_seg)


        PSNR_full, PSNR_tumor, PSNR_non_tumor, SSIM_full, SSIM_tumor, SSIM_non_tumor = compute_metrics(array_real, array_syn, array_seg, normalize=True)

        print(f'PSNR_full: {PSNR_full}, PSNR_tumor: {PSNR_tumor}, PSNR_non_tumor: {PSNR_non_tumor}, SSIM_full: {SSIM_full}, SSIM_tumor: {SSIM_tumor}, SSIM_non_tumor: {SSIM_non_tumor}')
        
        psnr_full_list.append(PSNR_full)
        psnr_tumor_list.append(PSNR_tumor)
        psnr_non_tumor_list.append(PSNR_non_tumor)
        ssim_full_list.append(SSIM_full)
        ssim_tumor_list.append(SSIM_tumor)
        ssim_non_tumor_list.append(SSIM_non_tumor)

        f = open(metric_txt, "a")
        f.write(f'{test_case}\t{PSNR_full}\t{PSNR_tumor}\t{PSNR_non_tumor}\t{SSIM_full}\t{SSIM_tumor}\t{SSIM_non_tumor}\t{MSE_full}\t{MSE_healthy}\t{MSE_tumor}')
        f.write('\n')
        f.close()
        
    # fold 별 평균
    # print(f'Average MSE_full: {np.mean(mse_full_list)}, Average MSE_tumor: {np.mean(mse_tumor_list)}, Average MSE_non_tumor: {np.mean(mse_non_tumor_list)}, Average PSNR_full: {np.mean(psnr_full_list)}, Average PSNR_tumor: {np.mean(psnr_tumor_list)}, Average PSNR_non_tumor: {np.mean(psnr_non_tumor_list)}, Average SSIM_full: {np.mean(ssim_full_list)}, Average SSIM_tumor: {np.mean(ssim_tumor_list)}, Average SSIM_non_tumor: {np.mean(ssim_non_tumor_list)}')    
    print(f'Average PSNR_full: {np.mean(psnr_full_list)}, Average PSNR_tumor: {np.mean(psnr_tumor_list)}, Average PSNR_non_tumor: {np.mean(psnr_non_tumor_list)}, Average SSIM_full: {np.mean(ssim_full_list)}, Average SSIM_tumor: {np.mean(ssim_tumor_list)}, Average SSIM_non_tumor: {np.mean(ssim_non_tumor_list)}')    
    
    f = open(metric_txt, "a")
    f.write(f'case_avg\t{np.mean(psnr_full_list)}\t{np.mean(psnr_tumor_list)}\t{np.mean(psnr_non_tumor_list)}\t{np.mean(ssim_full_list)}\t{np.mean(ssim_tumor_list)}\t{np.mean(ssim_non_tumor_list)}')
    f.write('\n')
    f.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # set start method to 'spawn'

    exp_name_list = [
                # ('BraSyn_3to1_real_t1c_fold0_finetune_H100', 't1c'),
                ('BraSyn_3to1_real_t1c_fold0_pretrain_H100', 't1c'),
                # ('BraSyn_3to1_real_t1n_fold0_finetune_H100', 't1n'),
                ('BraSyn_3to1_real_t1n_fold0_pretrain_H100', 't1n'),
                # ('BraSyn_3to1_real_t2w_fold0_finetune_H100', 't2w'),
                ('BraSyn_3to1_real_t2w_fold0_pretrain_H100', 't2w'),
                # ('BraSyn_3to1_real_t2f_fold0_finetune_H100', 't2f'),
                ('BraSyn_3to1_real_t2f_fold0_pretrain_H100', 't2f'),
                ]
    folds_list = [0]
    modality_list = ["t1c"]
    
    
    # for fold_num, missing_modality in zip(folds_list, modality_list):
    #     process_fold_modality(fold_num, missing_modality)
    
    # fold_modality_list = [(fold, modality) for fold in folds_list for modality in modality_list]
    fold_modality_list = [(fold, exp) for fold in folds_list for exp in exp_name_list]
    # with multiprocessing.Pool(8) as pool:
    #     pool.map(process_fold_modality, fold_modality_list)
    for fold, exp in fold_modality_list:
        exp_name, modality = exp
        process_fold_modality((fold, modality))