import nibabel as nib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

source_modality = ['']


####################################################################################################################
# inputs : nii_gz file | output dir | num_slices
# main functions
## make output dir
## minmax normalize to 0~255
## save each axial slices in png format
######################################################################################################################

def save_slices_as_png(nii_gz_file, output_dir, num_slices=100):
    # NIfTI 파일 읽기
    img = nib.load(nii_gz_file)
    data = img.get_fdata()
    
    # 출력 디렉터리 생성 (없는 경우)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    id = nii_gz_file.split('/')[-1].split('.')[0]
    
    # 이미지의 중심부터 num_slices 개수 만큼 슬라이스 추출
    start_slice = data.shape[2] // 2 - num_slices // 2
    end_slice = start_slice + num_slices

    data_min = data.min()
    data_max = data.max()

    # 슬라이스를 PNG 파일로 저장
    # for i in range(start_slice, end_slice):
    for i in range(data.shape[2]):
        unnormalized_data = data[:, :, i]
        normalized_data = (unnormalized_data - data_min) / (data_max - data_min) * 255
        normalized_data = normalized_data.astype(np.uint8)
        normalized_data = normalized_data.T
        normalized_data = np.flip(normalized_data, axis=0)
        image = Image.fromarray(normalized_data, 'L')  # 'L'은 그레이스케일 모드
        image.save(os.path.join(output_dir, id + f'slice_{str(i).zfill(3)}.png')) # on/off this code


##################################################################################################################
# You should do this
##################################################################################################################

with open(split_file, 'r') as f:
    splits = json.load(f)

dest_root = "../DATASET/"
src_folder = "../../Datasets/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

for fold in splits.keys():
    fold_dir = os.path.join(dest_root,fold)
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir,'train'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir,'test'), exist_ok=True)
    
    copy_folders(splits[fold]["train"], os.path.join(fold_dir, "train"))
    copy_folders(splits[fold]["test"], os.path.join(fold_dir, "test"))
    
def copy_folders(folder_list, destination):
    for i, folder_name in enumerate(folder_list,1):
        for modality in modality_list:
            src_dirr = os.path.join(src_folder, folder_name,folder_name + f'-{source_modality}.nii.gz')
            save_slices_as_png(src_dirr, destination)
            print(f'[{i}/{len(folder_list)}]-{modality} finished')
        print(f'{destination.split('/')[-1]}-[{i}/{len(folder_list)}] finished')
                
    
input_dir = '../../Datasets/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/fold_2/train'

mode_list = ['train', 'val', 'test'] # make selected dir
mode = 'val' # pick one among the mode_list

# one-to-one : set the src,target modality
source_modality = 't1c'
source_modality2 = 't1n'
source_modality3 = 't2f'
target_modality = 't2w'

# RUN
source_output_dir = f'../DATASET/ResViT/BraSyn/fold2/{mode}/{source_modality}'  # PNG 이미지를 저장할 디렉터리
source_output_dir2 = f'../DATASET/ResViT/BraSyn/fold2/{mode}/{source_modality2}'  # PNG 이미지를 저장할 디렉터리
source_output_dir3 = f'../DATASET/ResViT/BraSyn/fold2/{mode}/{source_modality3}'  # PNG 이미지를 저장할 디렉터리
target_output_dir = f'../DATASET/ResViT/BraSyn/fold2/{mode}/{target_modality}'  # PNG 이미지를 저장할 디렉터리


if mode == 'train': #25
    files = sorted(os.listdir(input_dir))[1:26]
elif mode == 'val': # 10
    files = sorted(os.listdir(input_dir))[101:111]
else: #mode = test # 20
    files = sorted(os.listdir(input_dir))[:]


for file in files:
    src_dirr = os.path.join(input_dir,file,file + f'-{source_modality}.nii.gz')
    save_slices_as_png(src_dirr, source_output_dir)
    print(src_dirr)

    src_dirr2 = os.path.join(input_dir,file,file + f'-{source_modality2}.nii.gz')
    save_slices_as_png(src_dirr2, source_output_dir2)
    print(src_dirr2)

    src_dirr3 = os.path.join(input_dir,file,file + f'-{source_modality3}.nii.gz')
    save_slices_as_png(src_dirr3, source_output_dir3)
    print(src_dirr3)

    target_dirr = os.path.join(input_dir,file,file + f'-{target_modality}.nii.gz')
    save_slices_as_png(target_dirr, target_output_dir)
    print(target_dirr)




