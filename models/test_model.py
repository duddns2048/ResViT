from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import random
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        assert(opt.phase == 'test')
        BaseModel.initialize(self, opt)
        opt.pre_trained_path = f'checkpoints/{opt.name}/latest_net_G.pth'
        if opt.dataset_mode == '1_to_1':
            input_nc=2
            output_nc=2
        elif opt.dataset_mode =='2_to_1':
            input_nc=3
            output_nc=3
        elif opt.dataset_mode =='3_to_1':
            input_nc=4
            output_nc=4
        elif opt.dataset_mode =='3_to_1_real':
            input_nc=3
            output_nc=1
        elif opt.dataset_mode =='uni':
            input_nc=4
            output_nc=4
        self.netG = networks.define_G(input_nc, output_nc, opt.ngf,
                                      opt.which_model_netG, 
                                      opt.vit_name, 
                                      opt.fineSize, 
                                      opt.pre_trained_path,
                                      opt.norm, 
                                      not opt.no_dropout,
                                      opt.init_type, 
                                      self.gpu_ids, 
                                      opt=opt)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        assert(self.opt.phase =='test')
        if self.opt.dataset_mode == 'uni':
            if self.opt.test_modality == 't1c':
                self.avail_fn = [0,1,1,1]
            elif self.opt.test_modality == 't1n':
                self.avail_fn = [1,0,1,1]
            elif self.opt.test_modality == 't2w':
                self.avail_fn = [1,1,0,1]
            elif self.opt.test_modality == 't2f':
                self.avail_fn = [1,1,1,0]
            missing_modality_index = self.avail_fn.index(0)
        elif self.opt.dataset_mode == "1_to_1":
            missing_modality_index = 1
        elif self.opt.dataset_mode == "2_to_1":
            missing_modality_index = 2
        elif self.opt.dataset_mode in ["3_to_1", '3_to_1_real'] :
            if self.opt.three_to_1_target == 't1c':
                self.avail_fn = [0,1,1,1]
            elif self.opt.three_to_1_target == 't1n':
                self.avail_fn = [1,0,1,1]
            elif self.opt.three_to_1_target == 't2w':
                self.avail_fn = [1,1,0,1]
            elif self.opt.three_to_1_target == 't2f':
                self.avail_fn = [1,1,1,0]
            self.missing_modality_index = self.avail_fn.index(0)
        else:
            raise('error')
        
        self.input_A = input['input_images']
        
        if self.opt.dataset_mode =='uni':
            self.input_A[:, missing_modality_index, :, :] = 0
        elif self.opt.dataset_mode =='3_to_1':
            self.input_A[:, missing_modality_index, :, :] = 0
        elif self.opt.dataset_mode =='3_to_1_real':
            self.input_A = torch.cat((self.input_A[:, :self.missing_modality_index, :, :], self.input_A[:, self.missing_modality_index+1:, :, :]), dim=1)
            
        if len(self.gpu_ids) > 0:
            self.input_A = self.input_A.cuda(self.gpu_ids[0], non_blocking=True)

    def test(self):
        # with torch.no_grad():
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
