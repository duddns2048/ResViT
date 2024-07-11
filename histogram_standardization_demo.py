import torch
import numpy as np
from pathlib import Path
import os
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
import torch
import numpy as np
import SimpleITK as sitk 
import os
import json
import multiprocessing
from tqdm import tqdm
from custom_histogram_standardization import CustomHistogramStandardization, denormalize
import time
from PIL import Image

# 모달리티 별 percentile clip 값
modality_clip_dict = {
    "t1n": 1360,
    "t1c": 3832,
    "t2w": 4845,
    "t2f": 1283
}


# Define evaluation Metrics
psnr = PeakSignalNoiseRatio(data_range=1.0) #because we normalize to 0-1
ssim = StructuralSimilarityIndexMeasure()

def save_thumbnail(img, denorm_option, case_id, modality, gt_max):
    os.makedirs(os.path.join('thumbnails', str(denorm_option), modality), exist_ok=True)
    
    img = np.clip(img, 0, gt_max)
    
    img = (img - 0) / (gt_max - 0)

    thumbnail_img = img[..., img.shape[-1] // 2][0]
    
    img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
    
    img_pil.save(os.path.join('thumbnails', str(denorm_option), modality, f'{case_id}-{modality}.png'))
    
    

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

    return output_tensor, v_max

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
        gt_image, gt_max = __percentile_clip(gt_image, p_min=0.5, p_max=99.5, strictlyPositive=True)
        prediction, _ = __percentile_clip(prediction, p_min=0.5, p_max=99.5, strictlyPositive=True)
    
    SSIM_full = ssim(preds=prediction, target=gt_image)

    return float(SSIM_full), gt_max



def transform_histogram_standardization():
    modality_list = ['t1n', 't1c', 't2f', 't2w']
    landmarks = {
        't1n': 'project/landmarks/t1n_landmarks_1_percent.npy',
        't1c': 'project/landmarks/t1c_landmarks_1_percent.npy',
        't2f': 'project/landmarks/t2f_landmarks_1_percent.npy',
        't2w': 'project/landmarks/t2w_landmarks_1_percent.npy',
    }

    histogram_transform = tio.Compose([
        CustomHistogramStandardization(landmarks, masking_method=lambda x: x > 0)
    ])

    data_dir = '/home/ubuntu/Datasets/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

    case_list = sorted(os.listdir(data_dir))
    case_list = [case for case in case_list if 'BraTS' in case]
    case_list = case_list[:100]

    subjects = []
    for case_id in case_list:
        subject = tio.Subject(
            t1n=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t1n.nii.gz')),
            t1c=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t1c.nii.gz')),
            t2w=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t2w.nii.gz')),
            t2f=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t2f.nii.gz')),
            seg=tio.LabelMap(os.path.join(data_dir, case_id, f'{case_id}-seg.nii.gz')),
            case_id=case_id
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    
    ssim_full_1_dict = {}
    ssim_full_2_dict = {}
    ssim_full_3_dict = {}
    ssim_full_4_dict = {}
    
    
    for i, sample in enumerate(tqdm(dataset)):
        case_id = sample.case_id
        
        t1c_ori = sample.t1c.data
        t1n_ori = sample.t1n.data
        t2f_ori = sample.t2f.data
        t2w_ori = sample.t2w.data
        mask = sample.seg.data
        
        standard = histogram_transform(sample)  
        
        t1c_norm = standard.t1c.data
        t1n_norm = standard.t1n.data
        t2f_norm = standard.t2f.data
        t2w_norm = standard.t2w.data

        t1c_linear_mapping = standard.t1c.linear_mapping
        t1n_linear_mapping = standard.t1n.linear_mapping
        t2f_linear_mapping = standard.t2f.linear_mapping
        t2w_linear_mapping = standard.t2w.linear_mapping
                
        # print execution time
        # start_time = time.time()
        # print(f'{case_id} start')
        
        # if one modality is missing
        for modality in modality_list:
            if modality not in ssim_full_1_dict:
                ssim_full_1_dict[modality] = []
                ssim_full_2_dict[modality] = []
                ssim_full_3_dict[modality] = []
                ssim_full_4_dict[modality] = []
                
            available_modalities = [m for m in modality_list if m != modality]
            
            # get average of other modalities' linear mapping
            fallback_linear_mapping = {}
            for key in t1c_linear_mapping.keys():
                # average of all modalities except for 'modality'
                fallback_linear_mapping[key] = 0.0
                for available_modality in available_modalities:
                    fallback_linear_mapping[key] += eval(f'{available_modality}_linear_mapping')[key]
                    fallback_linear_mapping[key] = fallback_linear_mapping[key] / len(available_modalities)
                
            """ 
                 strecthing with the percentile clip statistics
            """
            # 1. first clip normalized image to 0-100
            # and stretch to the original scale
            missing_modality_denorm = torch.clamp(eval(f'{modality}_norm'), 0, 100)
            missing_modality_denorm = missing_modality_denorm / 100
            missing_modality_denorm = missing_modality_denorm * modality_clip_dict[modality]
            ssim_full, gt_max = compute_metrics(eval(f'{modality}_ori').unsqueeze(0), missing_modality_denorm.unsqueeze(0), mask=mask, normalize=True)
            save_thumbnail(missing_modality_denorm.numpy(), 1, case_id, modality, gt_max)
            save_thumbnail(eval(f'{modality}_ori').numpy(), 'gt', case_id, f'{modality}-gt', gt_max)
            ssim_full_1_dict[modality].append(ssim_full)
            
            # 2. normalize to 0-1
            missing_modality_denorm = (eval(f'{modality}_norm') - eval(f'{modality}_norm').min()) / (eval(f'{modality}_norm').max() - eval(f'{modality}_norm').min())     
            missing_modality_denorm = missing_modality_denorm * modality_clip_dict[modality]
            ssim_full, gt_max = compute_metrics(eval(f'{modality}_ori').unsqueeze(0), missing_modality_denorm.unsqueeze(0), mask=mask, normalize=True)
            save_thumbnail(missing_modality_denorm.numpy(), 2, case_id, modality, gt_max)
            ssim_full_2_dict[modality].append(ssim_full)
                        
            

            
            """
                destandardize the missing modalisty using the average of the other modalities' linear mapping
            """            
            # 3.             
            missing_modality_denorm = denormalize(eval(f'{modality}_norm'), fallback_linear_mapping, mask=None)
            missing_modality_denorm = np.clip(missing_modality_denorm, 0, missing_modality_denorm.max())            
            ssim_full, gt_max = compute_metrics(eval(f'{modality}_ori').unsqueeze(0), missing_modality_denorm.unsqueeze(0), mask=mask, normalize=True)
            save_thumbnail(missing_modality_denorm.numpy(), 3, case_id, modality, gt_max)
            ssim_full_3_dict[modality].append(ssim_full)
            
            # 4.
            missing_modality_denorm = np.clip(eval(f'{modality}_norm'), 0, 100)
            missing_modality_denorm = denormalize(missing_modality_denorm, fallback_linear_mapping, mask=None)
            missing_modality_denorm = np.clip(missing_modality_denorm, 0, missing_modality_denorm.max())
            ssim_full, gt_max = compute_metrics(eval(f'{modality}_ori').unsqueeze(0), missing_modality_denorm.unsqueeze(0), mask=mask, normalize=True)
            save_thumbnail(missing_modality_denorm.numpy(), 4, case_id, modality, gt_max)
            ssim_full_4_dict[modality].append(ssim_full)
            
            # # print('ssim calculation start: ', time.time() - start_time)
            # # start_time = time.time()
            
            
            # ssim_tumor_dict[modality].append(ssim_tumor)
            # ssim_nontumor_dict[modality].append(ssim_nontumor)
            
            # print('modality done: ', time.time() - start_time)
            # start_time = time.time()


            
    for modality in modality_list:
        print(f'{modality} full 1: {np.mean(ssim_full_1_dict[modality])}')
        print(f'{modality} full 2: {np.mean(ssim_full_2_dict[modality])}')
        print(f'{modality} full 3: {np.mean(ssim_full_3_dict[modality])}')
        print(f'{modality} full 4: {np.mean(ssim_full_4_dict[modality])}')
        


def hard_norm_denorm():
    modality_list = ['t1n', 't1c', 't2f', 't2w']
    landmarks = {
        't1n': 't1n_landmarks_1_percent.npy',
        't1c': 't1c_landmarks_1_percent.npy',
        't2f': 't2f_landmarks_1_percent.npy',
        't2w': 't2w_landmarks_1_percent.npy',
    }

    histogram_transform = tio.Compose([
        # tio.CropOrPad((192, 192, 144)),
        CustomHistogramStandardization(landmarks, masking_method=lambda x: x > 0)
    ])

    data_dir = '/home/ubuntu/Datasets/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

    case_list = sorted(os.listdir(data_dir))
    case_list = [case for case in case_list if 'BraTS' in case]
    # case_list = case_list[:200]

    subjects = []
    for case_id in case_list:
        subject = tio.Subject(
            t1n=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t1n.nii.gz')),
            t1c=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t1c.nii.gz')),
            t2w=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t2w.nii.gz')),
            t2f=tio.ScalarImage(os.path.join(data_dir, case_id, f'{case_id}-t2f.nii.gz')),
            seg=tio.LabelMap(os.path.join(data_dir, case_id, f'{case_id}-seg.nii.gz')),
            case_id=case_id
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    
    ssim_full_dict = {}
    ssim_tumor_dict = {}
    ssim_nontumor_dict = {}
    
    for i, sample in enumerate(tqdm(dataset)):
        case_id = sample.case_id
        
        t1c_ori = sample.t1c.data
        t1n_ori = sample.t1n.data
        t2f_ori = sample.t2f.data
        t2w_ori = sample.t2w.data
        mask = sample.seg.data
        
        
        t1c_norm = (t1c_ori - t1c_ori.min()) / (t1c_ori.max() - t1c_ori.min())
        t1n_norm = (t1n_ori - t1n_ori.min()) / (t1n_ori.max() - t1n_ori.min())
        t2f_norm = (t2f_ori - t2f_ori.min()) / (t2f_ori.max() - t2f_ori.min())
        t2w_norm = (t2w_ori - t2w_ori.min()) / (t2w_ori.max() - t2w_ori.min())

        
        t1c_scale = t1c_ori.max() - t1c_ori.min()
        t1n_scale = t1n_ori.max() - t1n_ori.min()
        t2f_scale = t2f_ori.max() - t2f_ori.min()
        t2w_scale = t2w_ori.max() - t2w_ori.min()
        
        # if one modality is missing
        for modality in modality_list:
            if modality not in ssim_full_dict:
                ssim_full_dict[modality] = []
                ssim_tumor_dict[modality] = []
                ssim_nontumor_dict[modality] = []
                
            available_modalities = [m for m in modality_list if m != modality]
                        
            """
                destandardize with the other modalities' scale
            """            
            # 5.
            average_scale = 0.0
            for available_modality in available_modalities:
                average_scale += eval(f'{available_modality}_scale')
            average_scale = average_scale / len(available_modalities)
            
            missing_modality_denorm = eval(f'{modality}_norm') * average_scale
            
            missing_modality_denorm = missing_modality_denorm[np.newaxis, ...]
            ori_img = eval(f'{modality}_ori')[np.newaxis, ...]
            
            missing_modality_denorm = torch.from_numpy(missing_modality_denorm)
            ori_img = torch.from_numpy(ori_img)
                
            ssim_full = compute_metrics(ori_img, missing_modality_denorm, mask=mask, normalize=True)
            
            ssim_full_dict[modality].append(ssim_full)

            
    for modality in modality_list:
        print(f'{modality} full: {np.mean(ssim_full_dict[modality])}')        


    
if __name__ == '__main__':
    save_hard_normalized(0)