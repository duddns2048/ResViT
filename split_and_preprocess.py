import os
import json
import nibabel as nib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

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

    return output_tensor, v_min, v_max

def copy_folders(folder_list, destination):
    for i, folder_name in enumerate(folder_list,1):
        mode = destination.split('/')[-1] # train/val/test
        for modality in modality_list:
            output_dir = os.path.join(destination, modality)
            os.makedirs(output_dir, exist_ok=True)
            src_dirr = os.path.join(src_folder, folder_name,folder_name + f'-{modality}.nii.gz')
            save_slices(src_dirr, output_dir, norm = norm, file_format = file_format)
            print(f'[{i}/{len(folder_list)}]-{modality} finished')
        print(f'{mode}-[{i}/{len(folder_list)}] finished')


def save_slices(nii_gz_file, output_dir, num_slices=100, file_format = None, norm = None):
    # NIfTI 파일 읽기
    img = nib.load(nii_gz_file)
    data = img.get_fdata()
    
    # 출력 디렉터리 생성 (없는 경우)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    id = nii_gz_file.split('/')[-1].split('.')[0]
    
    # 이미지의 중심부터 num_slices 개수 만큼 슬라이스 추출
    # start_slice = data.shape[2] // 2 - num_slices // 2
    # end_slice = start_slice + num_slices
    if norm  == '255':
        data_min = data.min()
        data_max = data.max()
        normalized_data = (data - data_min) / (data_max - data_min) * 255
        
    elif norm == 'percentile_clip': # clip & normalize to [0:1]
        normalized_data, v_min, v_max = __percentile_clip(data, p_min=0.5, p_max=99.5, strictlyPositive=True)
        # print(f'v_min:{v_min}, v_max:{v_max}')
    else:
        raise Exception('==================Select correct norm policy!==================')
        
    # 슬라이스를 PNG 파일로 저장
    # for i in range(start_slice, end_slice):
    for i in range(normalized_data.shape[2]):
        normalized_slice_data = normalized_data[:,:,i]
        normalized_slice_data = normalized_slice_data.T
        normalized_slice_data = np.flip(normalized_slice_data, axis=0)
        if file_format == 'png':
            normalized_slice_data = normalized_slice_data.astype(np.uint8)
            image = Image.fromarray(normalized_slice_data, 'L')  # 'L'은 그레이스케일 모드
            image.save(os.path.join(output_dir, id + f'slice_{str(i).zfill(3)}.png')) # on/off this code
        elif file_format == 'npy':
            np.save(os.path.join(output_dir, id + f'_slice_{str(i).zfill(3)}.npy'), normalized_slice_data)
        
################
split_file = "../../Datasets/BraTS2023/case_ids/splits_brats.json"
modality_list = ['t1n', 't1c', 't2w', 't2f']
dest_root = "../DATASET/BRATS2023/2D_data_npy"
src_folder = "../../Datasets/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
norm = 'percentile_clip'
file_format = 'npy'
################

with open(split_file, 'r') as f:
    splits = json.load(f)

for fold in splits.keys():
    fold_dir = os.path.join(dest_root,fold)
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir,'train'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir,'val'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir,'test'), exist_ok=True)
    
    copy_folders(splits[fold]["train"], os.path.join(fold_dir, "train"))
    copy_folders(splits[fold]["test"][:20], os.path.join(fold_dir, "val"))
    copy_folders(splits[fold]["test"], os.path.join(fold_dir, "test"))
    
    break

