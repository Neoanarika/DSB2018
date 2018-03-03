from common import *

from net.resnet50_fpn_mask_single_shot.configuration import *
from net.lib.ssd.roi_align_pool_tf.module import RoIAlign as Crop

from net.lib.ssd.rpn_multi_nms    import *
from net.lib.ssd.rpn_multi_target import *
from net.lib.ssd.rpn_multi_loss   import *
from net.lib.ssd.mask_nms    import *
from net.lib.ssd.mask_target import *
from net.lib.ssd.mask_loss   import *


#############  resent50 pyramid feature net ##############################################################################

# class ConvBn2d(nn.Module):
#
#     def merge_bn(self):
#         raise NotImplementedError
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
#         super(ConvBn2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
#         self.bn   = nn.BatchNorm2d(out_channels)
#
#         if is_bn is False:
#             self.bn =None
#
#     def forward(self,x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         return x


## C layers ## ---------------------------

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1):
        super(BottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.bn1   = nn.BatchNorm2d(in_planes,eps = 2e-5)
        self.conv1 = nn.Conv2d(in_planes,     planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv2 = nn.Conv2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv3 = nn.Conv2d(   planes, out_planes, kernel_size=1, padding=0, stride=1, bias=False)

        if is_downsample:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=False)


    def forward(self, x):
        if self.is_downsample:
            x = F.relu(self.bn1(x))
            z = self.conv1(x)
            z = F.relu(self.bn2(z))
            z = self.conv2(z)
            z = F.relu(self.bn3(z))
            z = self.conv3(z)
            z += self.downsample(x)
        else:
            z = F.relu(self.bn1(x))
            z = self.conv1(z)
            z = F.relu(self.bn2(z))
            z = self.conv2(z)
            z = F.relu(self.bn3(z))
            z = self.conv3(z)
            z += x

        return z


def make_layer_c0(in_planes, out_planes):
    layers = [
        ##nn.BatchNorm2d(in_planes, eps = 2e-5),
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=1, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)



def make_layer_c(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(BottleneckBlock(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(BottleneckBlock(out_planes, planes, out_planes))

    return nn.Sequential(*layers)



## P layers ## ---------------------------

class LateralBlock(nn.Module):
    def __init__(self, c_planes, p_planes, out_planes ):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top     = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c , p):
        _,_,H,W = c.size()

        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2,mode='nearest')
        p = p[:,:,:H,:W] + c
        p = self.top(p)

        return p


# 2 ways to downsize
# def make_layer_p5(in_planes, out_planes):
#     layers = [
#         nn.ReLU(inplace=True),
#         nn.Conv2d( in_planes, out_planes, kernel_size=3, stride=2, padding=1)
#     ]
#     return nn.Sequential(*layers)

# def make_layer_p5(in_planes, out_planes):
#     layers = [
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#     ]
#     return nn.Sequential(*layers)


## resenet50 + pyramid  ##
##  - indexing is different from paper. Paper is 1-index. Ours is 0-index.
##
class FeatureNet(nn.Module):

    def __init__(self, cfg, in_channels, out_channels=256 ):
        super(FeatureNet, self).__init__()
        self.cfg=cfg

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)

        self.layer_c1 = make_layer_c(   64,  64,  128, num_blocks=2, stride=1)
        self.layer_c2 = make_layer_c(  128,  64,  256, num_blocks=3, stride=2)
        self.layer_c3 = make_layer_c(  256, 128,  512, num_blocks=4, stride=2)
        self.layer_c4 = make_layer_c(  512, 256, 1024, num_blocks=6, stride=2)
        self.layer_c5 = make_layer_c( 1024, 512, 2048, num_blocks=3, stride=2)

        # top-down
        self.layer_p5 = nn.Conv2d   (2048, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p4 = LateralBlock(1024, out_channels, out_channels)
        self.layer_p3 = LateralBlock( 512, out_channels, out_channels)
        self.layer_p2 = LateralBlock( 256, out_channels, out_channels)
        self.layer_p1 = LateralBlock( 128, out_channels, out_channels)


    def forward(self, x):
        #pass                         #; print('input ',   x.size())
        c0 = self.layer_c0 (x)       #; print('layer_c0 ',c0.size())
                                     #
                                     #
        c1 = self.layer_c1(c0)       #; print('layer_c1 ',c1.size())
        c2 = self.layer_c2(c1)       #; print('layer_c2 ',c2.size())
        c3 = self.layer_c3(c2)       #; print('layer_c3 ',c3.size())
        c4 = self.layer_c4(c3)       #; print('layer_c4 ',c4.size())
        c5 = self.layer_c5(c4)       #; print('layer_c5 ',c4.size())
                                     #
        p5 = self.layer_p5(c5)       #; print('layer_p5 ',p5.size())
        p4 = self.layer_p4(c4, p5)   #; print('layer_p4 ',p4.size())
        p3 = self.layer_p3(c3, p4)   #; print('layer_p3 ',p3.size())
        p2 = self.layer_p2(c2, p3)   #; print('layer_p2 ',p2.size())
        p1 = self.layer_p1(c1, p2)   #; print('layer_p1 ',p1.size())

        features = [p1,p2,p3,p4]
        assert(self.cfg.rpn_num_heads == len(features))

        return features


    #-----------------------------------------------------------------------
    def load_pretrain_file(self,pretrain_file, skip=[]):
        raise NotImplementedError
    def merge_bn(self):
        raise NotImplementedError



############# various head ##############################################################################################

class RpnMultiHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RpnMultiHead, self).__init__()

        self.num_classes = cfg.num_classes
        self.num_heads   = cfg.rpn_num_heads
        self.num_bases   = cfg.rpn_num_bases

        self.convs  = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.deltas = nn.ModuleList()
        for i in range(self.num_heads):
        	self.convs.append ( nn.Conv2d(in_channels, 256,                       kernel_size=3, padding=1) )
        	self.logits.append( nn.Conv2d(256, self.num_bases*self.num_classes,   kernel_size=3, padding=1) )
        	self.deltas.append( nn.Conv2d(256, self.num_bases*self.num_classes*4, kernel_size=3, padding=1) )


    def forward(self, fs):
        batch_size = len(fs[0])

        logits_flat = []
        probs_flat  = []
        deltas_flat = []
        for i in range(self.num_heads):  # apply multibox head to feature maps
            f = fs[i]
            f = F.dropout(f, p=0.5,training=self.training)
            f = F.relu(self.convs[i](f))
            logit = self.logits[i](f)
            delta = self.deltas[i](f)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes, 4)
            prob_flat  = F.softmax(logit_flat)  #softmax

            logits_flat.append(logit_flat)
            probs_flat.append (prob_flat )
            deltas_flat.append(delta_flat)


        logits_flat = torch.cat(logits_flat,1)
        probs_flat  = torch.cat(probs_flat,1)
        deltas_flat = torch.cat(deltas_flat,1)

        return logits_flat, probs_flat, deltas_flat


# https://qiita.com/yu4u/items/5cbe9db166a5d72f9eb8
#
# proposal i,x0,y0,x1,y1,score, label
# roi      i,x0,y0,x1,y1
# box        x0,y0,x1,y1

class CropRoi(nn.Module):
    def __init__(self, cfg, in_channels, out_channels ):
        super(CropRoi, self).__init__()
        self.num_heads = cfg.rpn_num_heads
        self.crop_size = cfg.crop_size
        self.scales = cfg.rpn_scales

        self.convs = nn.ModuleList()
        for i in range(self.num_heads):
            if self.scales[i] ==1:
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                ))
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.Upsample(scale_factor=self.scales[i],mode='bilinear'),
                ))

        self.crop =  Crop(self.crop_size, self.crop_size, 1)


    def forward(self, fs, proposals):

        features=[]
        for i in range(self.num_heads):
            f = fs[i]
            f = self.convs[i](f)
            features.append(f)
        features = torch.cat(features,1)


        rois  = proposals[:,0:5]
        crops = self.crop(features, rois)

        return crops



class MaskHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskHead, self).__init__()
        num_classes = cfg.num_classes

        self.conv1 = nn.Conv2d( in_channels,256, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(         256,256, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(         256,256, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(         256,256, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.ConvTranspose2d(256,256, kernel_size=4, padding=1, stride=2, bias=False)
        self.classify = nn.Conv2d( 256,num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, crops):
        x = F.relu(self.conv1(crops),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = F.relu(self.conv3(x),inplace=True)
        x = self.conv4(x)
        x = self.conv5(x)
        logits = self.classify(x)
        probs  = F.sigmoid(logits)

        return logits, probs

############# mask rcnn net ##############################################################################

class MaskSingleShotNet(nn.Module):

    def __init__(self, cfg):
        super(MaskSingleShotNet, self).__init__()
        self.version = 'net version \'mask-single-shot-resnet50-fpn\''
        self.cfg  = cfg
        self.mode = 'train'

        feature_channels = 256
        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.rpn_head  = RpnMultiHead(cfg,feature_channels)
        self.crop      = CropRoi (cfg,feature_channels, 64)
        self.mask_head = MaskHead(cfg,feature_channels)


    def forward(self, inputs, truth_boxes=None,  truth_labels=None, truth_instances=None ):
        cfg  = self.cfg
        mode = self.mode
        batch_size = len(inputs)

        #features
        features = data_parallel(self.feature_net, inputs)

        #rpn proposals
        self.rpn_logits_flat, self.rpn_probs_flat, self.rpn_deltas_flat = data_parallel(self.rpn_head, features)
        self.rpn_base, self.rpn_window = make_rpn_windows(cfg, features)
        self.rpn_proposals = rpn_nms(cfg, mode, inputs, self.rpn_probs_flat, self.rpn_deltas_flat, self.rpn_window)

        if mode in ['train', 'valid']:
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(cfg, inputs, self.rpn_window, truth_boxes, truth_labels )


        #segmentation
        self.mask_proposals = make_mask_proposals(cfg, mode, self.rpn_proposals)
        if self.mask_proposals is None:
            self.masks = make_empty_masks(cfg, mode, inputs)
        else:
            if mode in ['train', 'valid']:
                self.mask_proposals, self.mask_labels, self.mask_instances,   = \
                    make_mask_target(cfg, inputs, self.mask_proposals, truth_boxes, truth_labels, truth_instances)


            crops = self.crop(features, self.mask_proposals)
            self.mask_logits, self.mask_probs = data_parallel(self.mask_head, crops)
            self.masks = mask_nms(cfg, mode, inputs, self.mask_proposals, self.mask_probs) #<todo> better nms for mask


        return


    def loss(self, inputs, truth_boxes, truth_labels, truth_instances):
        cfg  = self.cfg
        # self.rpn_cls_loss  = Variable(torch.zeros((1))).cuda()
        # self.rpn_reg_loss  = Variable(torch.zeros((1))).cuda()
        # self.mask_cls_loss = Variable(torch.zeros((1))).cuda()

        self.rpn_cls_loss, self.rpn_reg_loss = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights)

        if self.mask_proposals is not None:
            self.mask_cls_loss  = \
                mask_loss( self.mask_logits, self.mask_labels, self.mask_instances )
        else:
            self.mask_cls_loss = Variable(torch.zeros((1))).cuda()

        #this balancing factor is by trial and error
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss  + 3*self.mask_cls_loss
        return self.total_loss


    #<todo> freeze bn for imagenet pretrain
    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        #raise NotImplementedError







# check #################################################################
def run_check_feature_net():

    batch_size = 4
    C, H, W = 3, 256, 256
    feature_channels = 128

    x = torch.randn(batch_size,C,H,W)
    inputs = Variable(x).cuda()

    cfg = Configuration()
    feature_net = FeatureNet(cfg, C, feature_channels).cuda()

    ps = feature_net(inputs)

    print('')
    num_heads = len(ps)
    for i in range(num_heads):
        p = ps[i]
        print(i, p.size())



def run_check_multi_rpn_head():

    batch_size  = 16
    in_channels = 128
    H,W = 256, 256
    num_heads = 4
    feature_heights = [ int(H//2**h) for h in range(num_heads) ]
    feature_widths  = [ int(W//2**h) for h in range(num_heads) ]

    fs = []
    for h,w in zip(feature_heights,feature_widths):
        f = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
        f = Variable(torch.from_numpy(f)).cuda()
        fs.append(f)


    cfg = Configuration()
    rpn_head = RpnMultiHead(cfg, in_channels).cuda()
    logits_flat, probs_flat, deltas_flat = rpn_head(fs)

    print('logits_flat ',logits_flat.size())
    print('probs_flat  ',probs_flat.size())
    print('deltas_flat ',deltas_flat.size())
    print('')



def run_check_crop_head():


    #feature maps
    batch_size  = 16
    in_channels = 128
    out_channels = 64
    H,W = 256, 256
    num_heads = 4
    feature_heights = [ int(H//2**h) for h in range(num_heads) ]
    feature_widths  = [ int(W//2**h) for h in range(num_heads) ]

    fs = []
    for h,w in zip(feature_heights,feature_widths):
        f = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
        f = Variable(torch.from_numpy(f)).cuda()
        fs.append(f)

    #proposal i,x0,y0,x1,y1,score, label
    proposals = []
    for i in range(num_heads):

        batch_size    = 4
        num_proposals = 8
        xs = np.random.randint(0,64,num_proposals)
        ys = np.random.randint(0,64,num_proposals)
        sizes  = np.random.randint(8,64,num_proposals)
        scores = np.random.uniform(0,1,num_proposals)

        proposal = np.zeros((num_proposals,7),np.float32)
        proposal[:,0] = i
        proposal[:,1] = xs
        proposal[:,2] = ys
        proposal[:,3] = xs+sizes
        proposal[:,4] = ys+sizes
        proposal[:,5] = scores
        proposal[:,6] = 1
        proposals.append(proposal)

    proposals = np.vstack(proposals)
    proposals = Variable(torch.from_numpy(proposals)).cuda()


    #--------------------------------------
    cfg      = Configuration()
    crop_net = CropRoi(cfg, in_channels, out_channels).cuda()
    crops    = crop_net(fs, proposals)

    print('crops', crops.size())
    #exit(0)

    crops     = crops.data.cpu().numpy()
    proposals = proposals.data.cpu().numpy()

    #for m in range(num_proposals):
    for m in range(8):
        crop     = crops[m]
        proposal = proposals[m]

        i,x0,y0,x1,y1,score,label = proposal

        print ('i=%d, x0=%3d, y0=%3d, x1=%3d, y1=%3d, score=%0.2f'%(i,x0,y0,x1,y1,score) )
        print (crop[0,0,:5] )
        print ('')




def run_check_mask_head():

    num_rois    = 100
    in_channels = 256
    crop_size   = 14

    crops = np.random.uniform(-1,1,size=(num_rois, in_channels, crop_size, crop_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert(crop_size==cfg.crop_size)

    mask_head = MaskHead(cfg, in_channels).cuda()
    logits, probs = mask_head(crops)

    print('logits ',logits.size())
    print('')



def run_check_mask_net():

    batch_size, C, H, W = 1, 3, 128,128
    feature_channels = 64
    inputs = np.random.uniform(-1,1,size=(batch_size, C, H, W)).astype(np.float32)
    inputs = Variable(torch.from_numpy(inputs)).cuda()

    cfg = Configuration()
    mask_net = MaskSingleShotNet(cfg).cuda()

    mask_net.set_mode('eval')
    mask_net(inputs)

    print('rpn_logits_flat ',mask_net.rpn_logits_flat.size())
    print('rpn_probs_flat  ',mask_net.rpn_probs_flat.size())
    print('rpn_deltas_flat ',mask_net.rpn_deltas_flat.size())
    print('')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_feature_net()
    # run_check_multi_rpn_head()
    # run_check_crop_head()
    run_check_mask_head()

    #run_check_mask_net()




