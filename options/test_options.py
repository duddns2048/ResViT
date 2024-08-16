from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--test_modality', type=str,  help='which modality to test?')
        self.isTrain = False
        # in the case of 1 to 1 
        self.parser.add_argument('--one_to_one_source', type=str, default=None, help='source modality when 1_to_1 mode')
        self.parser.add_argument('--one_to_one_target', type=str, default=None, help='target modality when 1_to_1 mode')
        # in the case of 3 to 1
        self.parser.add_argument('--three_to_1_target', type=str, default=None, help='target modality when 1_to_1 mode')

