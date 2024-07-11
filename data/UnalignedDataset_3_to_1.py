import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


class UnalignedDataset_3_to_1(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_src1 = os.path.join(opt.dataroot, opt.phase , 'src1')
        self.dir_src2 = os.path.join(opt.dataroot, opt.phase , 'src2')
        self.dir_src3 = os.path.join(opt.dataroot, opt.phase , 'src3')
        self.dir_target = os.path.join(opt.dataroot, opt.phase , 'target')

        # 이미지 파일이면 파일 경로 리스트에 담아서 반환
        self.src1_paths = make_dataset(self.dir_src1) 
        self.src2_paths = make_dataset(self.dir_src2)
        self.src3_paths = make_dataset(self.dir_src3)
        self.target_paths = make_dataset(self.dir_target)

        self.src1_paths = sorted(self.src1_paths)
        self.src2_paths = sorted(self.src2_paths)
        self.src3_paths = sorted(self.src3_paths)
        self.target_paths = sorted(self.target_paths)

        self.src1_size = len(self.src1_paths)
        self.src2_size = len(self.src2_paths)
        self.src3_size = len(self.src3_paths)
        self.target_size = len(self.target_paths)
        self.transform = get_transform(opt) # resize_and_crop, Totensor, Normalize

    def __getitem__(self, index):
        src1_path = self.src1_paths[index % self.src1_size]
        if self.opt.serial_batches:
            index_B = index % self.src2_size
        else:
            if self.opt.phase=='val':
                index_B = index % self.src2_size
            else:    
                index_B = random.randint(0, self.src2_size - 1)
        src2_path = self.src2_paths[index % self.src2_size]
        src3_path = self.src3_paths[index % self.src3_size]
        target_path = self.target_paths[index % self.target_size]
        #print('(src1, B) = (%d, %d)' % (index, index_B))
        src1_img = Image.open(src1_path).convert('RGB')
        src2_img = Image.open(src2_path).convert('RGB')
        src3_img = Image.open(src3_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        src1 = self.transform(src1_img) # resize_and_crop, Totensor, Normalize
        src2 = self.transform(src2_img)
        src3 = self.transform(src3_img)
        target = self.transform(target_img)

        tmp = src1[0, ...] * 0.299 + src1[1, ...] * 0.587 + src1[2, ...] * 0.114
        tmp = src1[1, ...] 
        src1 = tmp.unsqueeze(0)

        tmp = src2[0, ...] * 0.299 + src2[1, ...] * 0.587 + src2[2, ...] * 0.114
        tmp = src2[1, ...] 
        src2 = tmp.unsqueeze(0)

        tmp = src3[0, ...] * 0.299 + src3[1, ...] * 0.587 + src3[2, ...] * 0.114
        tmp = src3[1, ...] 
        src3 = tmp.unsqueeze(0)

        source_images = torch.cat([src1,src2,src3], dim=0)

        tmp = target[0, ...] * 0.299 + target[1, ...] * 0.587 + target[2, ...] * 0.114
        tmp = target[1, ...] 
        target = tmp.unsqueeze(0)

        return {'source_images': source_images, 'target': target,
                'src1_paths': src1_path,
                'src2_paths': src2_path,
                'src3_paths': src3_path, 
                'target_paths': target_path}

    def __len__(self):
        return max(self.src1_size, self.src2_size, self.src3_size, self.target_size)

    def name(self):
        return 'UnalignedDataset_3_to_1'
