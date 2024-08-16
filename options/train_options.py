from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        
        # static use
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')
        # self.parser.add_argument('--trans_lr_coef', type=float, default=1, help='initial learning rate for adam')
        
        # hyperparam
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        
        # visualization
        self.parser.add_argument('--validation_freq', type=int, default=1, help='frequency of epoch to conduct validation')
        self.parser.add_argument('--visdom_upload_freq', type=int, default=1, help='frequency of showing training results on screen')     
        self.parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        
        # continue train
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        
        # finetune
        self.parser.add_argument('--pre_trained_path', type=str,default=None,help='path to the pre-trained resnet architecture')
        self.parser.add_argument('--pre_trained_resnet', type=int, default=0,help='Pre-trained residual CNNs or not')
        
        # options
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--randomsampler', type=int, default=0, help='if you want to use random sampler')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--fold', type=int, default=0, help='fold num')
        self.parser.add_argument('--G_only_L1', type=int, default=0, help='if specified, G_loss = L1 | else G_loss = GAN_loss + L1_loss')
        self.parser.add_argument('--num_cases', type=int, default=25, help='if you wnat to learn just few cases')

        # # 1 to 1
        self.parser.add_argument('--one_to_one_source', type=str, default=None, help='source modality when 1_to_1 mode')
        self.parser.add_argument('--one_to_one_target', type=str, default=None, help='target modality when 1_to_1 mode')

        # # 3 to 1
        self.parser.add_argument('--three_to_1_target', type=str, default=None, help='target modality when 1_to_1 mode')
        self.isTrain = True
        