import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
import random
import json

class UnalignedDataset_unified(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        self.dir_modality1 = os.path.join(opt.dataroot, opt.phase , 't1c') # t1c t1n t2w t2f
        self.dir_modality2 = os.path.join(opt.dataroot, opt.phase , 't1n')
        self.dir_modality3 = os.path.join(opt.dataroot, opt.phase , 't2w')
        self.dir_modality4 = os.path.join(opt.dataroot, opt.phase , 't2f')

        # 이미지 파일이면 파일 경로 리스트에 담아서 반환
        
        self.modality1_paths = make_dataset(self.dir_modality1) 
        self.modality2_paths = make_dataset(self.dir_modality2)
        self.modality3_paths = make_dataset(self.dir_modality3) 
        self.modality4_paths = make_dataset(self.dir_modality4) 

        self.modality1_paths = sorted(self.modality1_paths)
        self.modality2_paths = sorted(self.modality2_paths)
        self.modality3_paths = sorted(self.modality3_paths)
        self.modality4_paths = sorted(self.modality4_paths)
        if self.opt.phase == 'train':
            self.modality1_paths = self.modality1_paths[:3875]
            self.modality2_paths = self.modality2_paths[:3875]
            self.modality3_paths = self.modality3_paths[:3875]
            self.modality4_paths = self.modality4_paths[:3875]
        
        self.modality1_size = len(self.modality1_paths)
        self.modality2_size = len(self.modality2_paths)
        self.modality3_size = len(self.modality3_paths)
        self.modality4_size = len(self.modality4_paths)
        self.transform = get_transform(opt) # resize_and_crop, Totensor, Normalize(0.5, 0.5, 0.5)

    def __getitem__(self, index):

        # self.avail_fn = [1,1,1,1] # t1c t1n t2w t2f
        # random_number = random.randint(0, 3)
        # self.avail_fn[random_number] = 0

        # if self.opt.phase =='test':
        #     if self.opt.test_modality == 't1n':
        #         self.avail_fn = [0,1,1,1]
        #     elif self.opt.test_modality == 't1c':
        #         self.avail_fn = [1,0,1,1]
        #     elif self.opt.test_modality == 't2w':
        #         self.avail_fn = [1,1,0,1]
        #     elif self.opt.test_modality == 't2f':
        #         self.avail_fn = [1,1,1,0]

        modality1_path = self.modality1_paths[index % self.modality1_size]
        modality2_path = self.modality2_paths[index % self.modality2_size]
        modality3_path = self.modality3_paths[index % self.modality3_size]
        modality4_path = self.modality4_paths[index % self.modality4_size]
        #print('(modality1, B) = (%d, %d)' % (index, index_B))
        if modality1_path.endswith(".npy"):
            modality1_img = Image.fromarray(np.expand_dims(np.load(modality1_path).astype(np.float32),axis=0))
            modality2_img = Image.fromarray(np.expand_dims(np.load(modality2_path).astype(np.float32),axis=0))
            modality3_img = Image.fromarray(np.expand_dims(np.load(modality3_path).astype(np.float32),axis=0))
            modality4_img = Image.fromarray(np.expand_dims(np.load(modality4_path).astype(np.float32),axis=0))
            
            # modality1_img = np.stack([modality1_img]*3, axis=-1)
            # modality2_img = np.stack([modality2_img]*3, axis=-1)
            # modality3_img = np.stack([modality3_img]*3, axis=-1)
            # modality4_img = np.stack([modality4_img]*3, axis=-1)
            
            # modality1_img = Image.fromarray(modality1_img).convert('RGB')
            # modality2_img = Image.fromarray(modality2_img).convert('RGB')
            # modality3_img = Image.fromarray(modality3_img).convert('RGB')
            # modality4_img = Image.fromarray(modality4_img).convert('RGB')
            
            
        else:
            # modality1_img = Image.open(modality1_path)
            # modality2_img = Image.open(modality2_path)
            # modality3_img = Image.open(modality3_path)
            # modality4_img = Image.open(modality4_path)
            
            modality1_img = Image.open(modality1_path).convert('RGB')
            modality2_img = Image.open(modality2_path).convert('RGB')
            modality3_img = Image.open(modality3_path).convert('RGB')
            modality4_img = Image.open(modality4_path).convert('RGB')
        
        

        modality1 = self.transform(modality1_img) # resize_and_crop, Totensor, Normalize
        modality2 = self.transform(modality2_img)
        modality3 = self.transform(modality3_img)
        modality4 = self.transform(modality4_img)

        tmp = modality1[1, ...] 
        modality1 = tmp.unsqueeze(0)

        tmp = modality2[1, ...] 
        modality2 = tmp.unsqueeze(0)

        tmp = modality3[1, ...] 
        modality3 = tmp.unsqueeze(0)

        tmp = modality4[1, ...] 
        modality4 = tmp.unsqueeze(0)

        # input_images = torch.cat([ modality1 if self.avail_fn[0]== 1 else torch.zeros_like(modality1),
        #                             modality2 if self.avail_fn[1]== 1 else torch.zeros_like(modality2),
        #                             modality3 if self.avail_fn[2]== 1 else torch.zeros_like(modality3),
        #                             modality4 if self.avail_fn[3]== 1 else torch.zeros_like(modality4),
        #                             ], dim=0)
        
        input_images = torch.cat([modality1, 
                                    modality2,
                                    modality3,
                                    modality4,
                                    ], dim=0)

        output_images = torch.cat([modality1, 
                                    modality2,
                                    modality3,
                                    modality4,
                                    ], dim=0)
        
        # tmp = modality4[0, ...] * 0.299 + modality4[1, ...] * 0.587 + modality4[2, ...] * 0.114
        # tmp = modality4[1, ...] 
        # modality4 = tmp.unsqueeze(0)

        return {'input_images': input_images, 'output_images': output_images,
                'modality1_paths': modality1_path,
                'modality2_paths': modality2_path,
                'modality3_paths': modality3_path,
                'modality4_paths': modality4_path,
                }

    def __len__(self):
        return max(self.modality1_size, self.modality2_size, self.modality3_size, self.modality4_size)

    def name(self):
        return 'UnalignedDataset_unified'
    
    
class UnalignedDataset_unified2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        split_file = '/home/ubuntu/youngwoon/DATASET/splits_brats.json'
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if self.opt.phase =='train':
            self.case_names = splits[f'fold_{opt.fold}']["train"]
        elif self.opt.phase == 'val':
            self.case_names = splits[f'fold_{opt.fold}']["test"][:10]
        elif self.opt.phase =='test':
            self.case_names = splits[f'fold_{opt.fold}']["test"]
        
        self.modality_paths = os.path.join(opt.dataroot, opt.phase) # t1c t1n t2w t2f
        self.modality_size = len(self.case_names)
        self.transform = get_transform(opt) # resize_and_crop, Totensor, Normalize(0.5, 0.5, 0.5)

    def __getitem__(self, index):
        random_integer = random.randint(0, 154)
        random_integer = str(random_integer).zfill(3)
        
        if self.opt.phase =='test':
            if self.opt.test_modality == 't1c':
                self.avail_fn = [0,1,1,1]
            elif self.opt.test_modality == 't1n':
                self.avail_fn = [1,0,1,1]
            elif self.opt.test_modality == 't2w':
                self.avail_fn = [1,1,0,1]
            elif self.opt.test_modality == 't2f':
                self.avail_fn = [1,1,1,0]
        else:
            self.avail_fn = [1, 1, 1, 0]
            random.shuffle(self.avail_fn)

        modality1_path = os.path.join(self.modality_paths, 't1c', self.case_names[index] + '-t1cslice_' + random_integer + '.png')
        modality2_path = os.path.join(self.modality_paths, 't1n', self.case_names[index] + '-t1nslice_' + random_integer + '.png')
        modality3_path = os.path.join(self.modality_paths, 't2w', self.case_names[index] + '-t2wslice_' + random_integer + '.png')
        modality4_path = os.path.join(self.modality_paths, 't2f', self.case_names[index] + '-t2fslice_' + random_integer + '.png')
        #print('(modality1, B) = (%d, %d)' % (index, index_B))
        modality1_img = Image.open(modality1_path).convert('RGB')
        modality2_img = Image.open(modality2_path).convert('RGB')
        modality3_img = Image.open(modality3_path).convert('RGB')
        modality4_img = Image.open(modality4_path).convert('RGB')

        modality1 = self.transform(modality1_img) # resize_and_crop, Totensor, Normalize
        modality2 = self.transform(modality2_img)
        modality3 = self.transform(modality3_img)
        modality4 = self.transform(modality4_img)

        tmp = modality1[1, ...] 
        modality1 = tmp.unsqueeze(0)

        tmp = modality2[1, ...] 
        modality2 = tmp.unsqueeze(0)

        tmp = modality3[1, ...] 
        modality3 = tmp.unsqueeze(0)

        tmp = modality4[1, ...] 
        modality4 = tmp.unsqueeze(0)

        # input_images = torch.cat([ modality1 if self.avail_fn[0]== 1 else torch.zeros_like(modality1),
        #                             modality2 if self.avail_fn[1]== 1 else torch.zeros_like(modality2),
        #                             modality3 if self.avail_fn[2]== 1 else torch.zeros_like(modality3),
        #                             modality4 if self.avail_fn[3]== 1 else torch.zeros_like(modality4),
        #                             ], dim=0)
        
        input_images = torch.cat([modality1, 
                                    modality2,
                                    modality3,
                                    modality4,
                                    ], dim=0)

        output_images = torch.cat([modality1, 
                                    modality2,
                                    modality3,
                                    modality4,
                                    ], dim=0)
        
        # tmp = modality4[0, ...] * 0.299 + modality4[1, ...] * 0.587 + modality4[2, ...] * 0.114
        # tmp = modality4[1, ...] 
        # modality4 = tmp.unsqueeze(0)

        return {'input_images': input_images, 'output_images': output_images,
                'modality1_paths': modality1_path,
                'modality2_paths': modality2_path,
                'modality3_paths': modality3_path,
                'modality4_paths': modality4_path,
                'avail_fn': self.avail_fn }

    def __len__(self):
        if self.opt.phase == 'train':
            length = 1001
        elif self.opt.phase == 'val':
            length = 10
        elif self.opt.phase =='test':
            length = 250
        return self.modality_size

    def name(self):
        return 'UnalignedDataset_unified'
