from common import *
import configparser

class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version='configuration version \'mask-single-shot-resnet50-fpn kaggle\''

        #net
        self.num_classes = 2 #include background class

        #multi-rpn
        self.rpn_base_sizes         = [ 8, 16, 32, 64 ] #diameter
        self.rpn_scales             = [ 1,  2,  4,  8 ]
        self.rpn_base_apsect_ratios = [
            #[1, ],
            [1 ],
            [1, 0.5,  2],
            [1, 0.5,  2],
            [1, 0.5,  2],
        ]

        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low  = 0.7

        self.rpn_train_nms_pre_score_threshold = 0.6
        self.rpn_train_nms_overlap_threshold   = 0.8  #higher for more proposals for mask training
        self.rpn_train_nms_min_size  = 5

        self.rpn_test_nms_pre_score_threshold = 0.8
        self.rpn_test_nms_overlap_threshold   = 0.5
        self.rpn_test_nms_min_size  = 5

        #crop
        self.crop_size  = 14

        #mask
        self.mask_size  = 28
        self.mask_train_batch_size  = 32 #per image
        self.mask_train_min_size  = 5
        self.mask_train_fg_thresh_low = 0.5

        self.mask_test_threshold     = self.rpn_test_nms_pre_score_threshold
        self.mask_test_nms_threshold = 0.5




    #-------------------------------------------------------------------------------------------------------
    def __repr__(self):
        d = self.__dict__.copy()
        str=''
        for k, v in d.items():
            str +=   '%32s = %s\n' % (k,v)

        return str


    def save(self, file):
        d = self.__dict__.copy()
        config = configparser.ConfigParser()
        config['all'] = d
        with open(file, 'w') as f:
            config.write(f)


    def load(self, file):
        # config = configparser.ConfigParser()
        # config.read(file)
        #
        # d = config['all']
        # self.num_classes     = eval(d['num_classes'])
        # self.multi_num_heads = eval(d['multi_num_heads'])

        raise NotImplementedError
