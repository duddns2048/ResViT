import torch.utils.data
from torch.utils.data import RandomSampler
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode == 'unaligned': # 1 to 1
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == '1_to_1':
        if opt.phase =='train':
            from data.single_dataset import SingleDataset
            dataset = SingleDataset()
        elif (opt.phase == 'val') or (opt.phase == 'test'):
            from data.single_dataset import SingleDataset
            dataset = SingleDataset()
    elif opt.dataset_mode == '2_to_1':
        from data.UnalignedDataset_2_to_1 import UnalignedDataset_2_to_1
        dataset = UnalignedDataset_2_to_1()
    elif opt.dataset_mode == '3_to_1':
        from data.UnalignedDataset_3_to_1 import UnalignedDataset_3_to_1
        dataset = UnalignedDataset_3_to_1()
    elif opt.dataset_mode == '3_to_1_real':
        from data.UnalignedDataset_3_to_1 import UnalignedDataset_3_to_1_real
        dataset = UnalignedDataset_3_to_1_real()
    elif opt.dataset_mode == 'uni':
        if opt.num_cases:
            from data.UnalignedDataset_unified import UnalignedDataset_unified_fewcases
            dataset = UnalignedDataset_unified_fewcases()
        else:
            from data.UnalignedDataset_unified import UnalignedDataset_unified_randomsample
            dataset = UnalignedDataset_unified_randomsample()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)  
        print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
         
    return dataset 
   



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt): 
        
        BaseDataLoader.initialize(self, opt)  #TODO 이거해야됨? ㅇㅇ 그래야 self.opt가 활성화됨
        self.dataset = CreateDataset(opt)
        if opt.phase == 'train' and opt.randomsampler:
            sampler = RandomSampler(self.dataset, replacement=False, num_samples=15000)
        else:
            sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler = sampler,
            batch_size=opt.batchSize,
            # shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
