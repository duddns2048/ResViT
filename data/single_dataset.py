import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import json
import random
import torch

class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        self.dir_modality1 = os.path.join(opt.dataroot, opt.phase , self.opt.one_to_one_source) # t1c t1n t2w t2f
        self.dir_modality2 = os.path.join(opt.dataroot, opt.phase , self.opt.one_to_one_target)

        self.modality1_paths = make_dataset(self.dir_modality1) 
        self.modality2_paths = make_dataset(self.dir_modality2)

        self.modality1_paths = sorted(self.modality1_paths)
        self.modality2_paths = sorted(self.modality2_paths)
        if self.opt.phase == 'train':
            self.modality1_paths = self.modality1_paths[:3875]
            self.modality2_paths = self.modality2_paths[:3875]
        
        self.modality1_size = len(self.modality1_paths)
        self.modality2_size = len(self.modality2_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        modality1_path = self.modality1_paths[index]
        modality2_path = self.modality2_paths[index]
        
        modality1_img = Image.open(modality1_path).convert('RGB')
        modality2_img = Image.open(modality2_path).convert('RGB')
        
        modality1 = self.transform(modality1_img) # resize_and_crop, Totensor, Normalize
        modality2 = self.transform(modality2_img)
        
        tmp = modality1[1, ...] 
        modality1 = tmp.unsqueeze(0)
        
        tmp = modality2[1, ...] 
        modality2 = tmp.unsqueeze(0)
        
        input_images = torch.cat([modality1,
                                  modality2,
                                  ], dim=0)
        output_images = torch.cat([modality1,
                                  modality2,
                                  ], dim=0)

   

        return {'input_images': input_images, 
                'output_images': output_images,
                'modality_paths': modality1_path}

    def __len__(self):
        return self.modality1_size

    def name(self):
        return 'SingleImageDataset'
    
class SingleDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        self.modality_paths = os.path.join(opt.dataroot, opt.phase)
        
        split_file = '/home/ubuntu/youngwoon/DATASET/splits_brats.json'
        
        with open(split_file,'r') as f:
            splits = json.load(f)
            
        if opt.phase =='train':
            self.cases = splits[f'fold_{opt.fold}']['train']
        elif opt.phase =='val':
            self.cases = splits[f'fold_{opt.fold}']['test'][:10]
        elif opt.phase =='test':
            self.cases = splits[f'fold_{opt.fold}']['test']
        
        self.data_size = len(self.cases)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        random_integer = random.randint(0, 154)
        random_integer = str(random_integer).zfill(3)
        
        if self.opt.one_to_one_source == self.opt.one_to_one_target:
            raise('source and target is same!')
        input_image  = os.path.join(self.modality_paths,self.opt.one_to_one_source,self.cases[index]+f'-{self.opt.one_to_one_source}'+'slice_'+random_integer+'.png')
        output_image = os.path.join(self.modality_paths,self.opt.one_to_one_target,self.cases[index]+f'-{self.opt.one_to_one_target}'+'slice_'+random_integer+'.png')
        
        source_img = Image.open(input_image).convert('RGB')
        target_img = Image.open(output_image).convert('RGB')
        
        source = self.transform(source_img) # resize_and_crop, Totensor, Normalize
        target = self.transform(target_img)

        tmp = source[1, ...] 
        source = tmp.unsqueeze(0)

        tmp = target[1, ...] 
        target = tmp.unsqueeze(0)
        
        input_images = torch.cat([source,
                                  target,
                                  ], dim=0)

        output_images = torch.cat([source,
                                  target,
                                  ], dim=0) 

        return {'input_images': input_images, 
                'output_images': output_images}

    def __len__(self):
        return self.data_size

    def name(self):
        return 'SingleImageDataset'

