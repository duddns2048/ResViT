import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import random

class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define network
        if self.opt.dataset_mode =='3_to_1':
            G_input_c = 4
            G_output_c = 4
        elif self.opt.dataset_mode =='3_to_1_real':
            G_input_c = 3
            G_output_c = 1
        self.netG = networks.define_G(G_input_c, G_output_c, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet, opt=opt)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(G_input_c+1, opt.ndf,
                                          opt.which_model_netD,opt.vit_name,opt.fineSize,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
    # if self.opt.dataset_mode =='3_to_1':
        if self.opt.three_to_1_target == 't1n':
            self.avail_fn = [0,1,1,1]
        elif self.opt.three_to_1_target == 't1c':
            self.avail_fn = [1,0,1,1]
        elif self.opt.three_to_1_target == 't2w':
            self.avail_fn = [1,1,0,1]
        elif self.opt.three_to_1_target == 't2f':
            self.avail_fn = [1,1,1,0]
        self.missing_modality_index = self.avail_fn.index(0)
            
        self.input_A = input['input_images']
        self.input_B = input['output_images']
        
        if self.opt.dataset_mode =='3_to_1':
            self.input_A[:, self.missing_modality_index, :, :] = 0

        elif self.opt.dataset_mode =='3_to_1_real':
            self.input_A = torch.cat((self.input_A[:, :self.missing_modality_index, :, :], self.input_A[:, self.missing_modality_index+1:, :, :]), dim=1)
            self.input_B = self.input_B[:,self.missing_modality_index:self.missing_modality_index+1,:,:]
        else:
            raise Exception('dataset_mode is wrong!')
                
        
        
        
        
        if len(self.gpu_ids) > 0:
            self.input_A = self.input_A.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_B = self.input_B.cuda(self.gpu_ids[0], non_blocking=True)
        
        
        self.image_paths = input['input_images']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B= self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # src modality 2개와 fake target 1개 cat하여 D에 넣기
        if self.opt.dataset_mode =='3_to_1':
            fake_AB = torch.cat((self.real_A, self.fake_B[:,self.missing_modality_index:self.missing_modality_index+1,:,:]), 1)
        elif self.opt.dataset_mode =='3_to_1_real':
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) #
        # Real
        if self.opt.dataset_mode =='3_to_1':
            real_AB = torch.cat((self.real_A, self.real_B[:,self.missing_modality_index:self.missing_modality_index+1,:,:]), 1)
        elif self.opt.dataset_mode =='3_to_1_real':
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.dataset_mode =='3_to_1':
            fake_AB = torch.cat((self.real_A, self.fake_B[:,self.missing_modality_index:self.missing_modality_index+1,:,:]), 1)
        elif self.opt.dataset_mode =='3_to_1_real':    
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv # lambda_adv=1
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        if self.opt.G_only_L1:
            self.loss_G = self.loss_G_L1*1
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1*1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())

                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
