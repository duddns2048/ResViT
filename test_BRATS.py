import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.transform import resize


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    old_case_name = ''

    # test
    volume_list = []
    for i, data in enumerate(dataset):

        if opt.test_modality == 't1c':
            cur_case_name = data['modality1_paths'][0].split('/')[-1][:23]
        elif opt.test_modality == 't1n':
            cur_case_name = data['modality2_paths'][0].split('/')[-1][:23]
        elif opt.test_modality == 't2w':
            cur_case_name = data['modality3_paths'][0].split('/')[-1][:23]
        elif opt.test_modality == 't2f':
            cur_case_name = data['modality4_paths'][0].split('/')[-1][:23]

        nii_image = nib.load(f'../DATASET/BRATS2024/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/fold_1/test/{data['modality4_paths'][0].split('/')[-1][:19]}/{data['modality4_paths'][0].split('/')[-1][:19]}-t1c.nii.gz')
        affine_matrix = nii_image.affine
        # affine_matrix = np.array([[ -1.,  -0.,  -0.,   0.], [ -0.,  -1.,  -0., 239.],  [  0.,   0.,   1.,   0.], [  0.,   0.,   0.,   1.]])
        header = nii_image.header
        voxel_shape = header.get_data_shape()
        # voxel_shape = (240,240,155)

        
        model.set_input(data) # A이미지, 경로 세팅
        model.test() # realA와 fakeB=G(realA) 생성
        visuals = model.get_current_visuals() # 텐서를 넘파이 이미지로 바꿔 딕셔너리 반환
        img_path = model.get_image_paths()
        # image_pil = Image.fromarray(visuals['fake_B'])
        # image_pil = Image.fromarray(visuals['fake_B'].squeeze())
        # image_pil.save(f'./results/{opt.name}/{data['modality4_paths'][0].split('/')[-1]}')

        if opt.test_modality == 't1c':
            volume_list.append(np.expand_dims(visuals['fake_B'][...,0],axis=-1))
        elif opt.test_modality == 't1n':
            volume_list.append(np.expand_dims(visuals['fake_B'][...,1],axis=-1))
        elif opt.test_modality == 't2w':
            volume_list.append(np.expand_dims(visuals['fake_B'][...,2],axis=-1))
        elif opt.test_modality == 't2f':
            volume_list.append(np.expand_dims(visuals['fake_B'][...,3],axis=-1))
            
        if (i+1)%155==0:
            print(f'[{(i+1)//155}/250]')
            volume = np.concatenate(volume_list, axis=2)
            volume = volume.transpose(1,0,2)
            volume = volume[:,::-1,:]
            resampled_image = resize(volume, voxel_shape, mode='constant', anti_aliasing=True)
            nii_image = nib.Nifti1Image(resampled_image, affine=affine_matrix)

            output_dir = f'./checkpoints/{opt.name}/results/fold1/{cur_case_name[:-4]}'
            if not os.path.exists(output_dir):
                os.makedirs(os.path.join(output_dir))
            nib.save(nii_image, os.path.join(output_dir,f'{cur_case_name}.nii.gz'))

            volume_list = []

        
        if opt.test_modality == 't1c':
            num = 1
        elif opt.test_modality == 't1n':
            num = 2
        elif opt.test_modality == 't2w':
            num = 3
        elif opt.test_modality == 't2f':
            num = 4
        print(data[f'modality{num}_paths'][0].split('/')[-1])
        
