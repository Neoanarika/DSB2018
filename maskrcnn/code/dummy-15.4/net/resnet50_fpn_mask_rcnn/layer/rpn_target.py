# reference:  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
from common import *
from utility.draw import *


from net.resnet50_fpn_mask_rcnn.layer.rpn_multi_nms import *

## debug and draw #############################################################
def normalize(data):
    data = (data-data.min())/(data.max()-data.min())
    return data


def unflat_to_c3(data, num_scales, num_bases, H, W):
    datas=[]

    start = 0
    for l in range(num_scales):
        h,w = int(H/2**l), int(W/2**l)
        c = num_bases[l]

        size = h*w*c
        d = data[start:start+size].reshape(h,w,c)
        start = start+size

        if c==1:
            d = d*np.array([1,1,1])
        elif c==3:
            pass
        else:
            raise NotImplementedError

        datas.append(d)

    return datas

## target #############################################################

# cpu version
def make_one_rpn_target(cfg, input, window, truth_box, truth_label):


    num_window = len(window)
    label         = np.zeros((num_window, ), np.float32)
    label_assign  = np.zeros((num_window, ), np.int32)
    label_weight  = np.zeros((num_window, ), np.float32)
    target        = np.zeros((num_window,4), np.float32)
    target_weight = np.zeros((num_window, ), np.float32)


    num_truth_box = len(truth_box)
    if num_truth_box != 0:

        # classification ---------------------------------------

        # bg
        overlap        = cython_box_overlap(window, truth_box)
        argmax_overlap = np.argmax(overlap,1)
        max_overlap    = overlap[np.arange(num_window),argmax_overlap]

        bg_index = max_overlap <  cfg.rpn_train_bg_thresh_high
        label[bg_index] = 0
        label_weight[bg_index] = 1


        # fg
        fg_index = max_overlap >=  cfg.rpn_train_fg_thresh_low
        label[fg_index] = 1  #<todo> extend to multi-class ... need to modify regression below too
        label_weight[fg_index] = 1
        label_assign[...] = argmax_overlap









    #if 0:
        # window
        _,height,width = input.size()

        allowed_border=0
        valid_window_index = np.where(
            (window[:,0] >= allowed_border)    & \
            (window[:,1] >= allowed_border)    & \
            (window[:,2] <= width-1  - allowed_border) & \
            (window[:,3] <= height-1 - allowed_border))[0]
        valid_window = window[valid_window_index]
        num_valid_window = len(valid_window)

        # truth_box
        valid_truth_box_index = np.where(truth_label>=0)[0]
        valid_truth_box       = truth_box[valid_truth_box_index]
        num_valid_truth_box   = len(valid_truth_box)
        num_truth_box         = len(truth_box)

        # classification ---------------------------------------

        # bg
        overlap        = cython_box_overlap(valid_window, truth_box)
        argmax_overlap = np.argmax(overlap,1)
        max_overlap    = overlap[np.arange(num_valid_window),argmax_overlap]

        bg_ind = np.where(max_overlap <  cfg.rpn_train_bg_thresh_high)[0]
        if (bg_ind.any()):
            bg_ind = valid_window_index[bg_ind]
            label[bg_ind]=0


        # fg
        overlap        = cython_box_overlap(valid_window, valid_truth_box)
        argmax_overlap = np.argmax(overlap,1)
        max_overlap    = overlap[np.arange(num_valid_window),argmax_overlap]

        fg_ind = np.where((max_overlap >= cfg.rpn_train_fg_thresh_low))[0]
        if (fg_ind.any()):
            a = argmax_overlap[fg_ind]
            fg_ind = valid_window_index[fg_ind]
            label [fg_ind] = 1
            label_assign[fg_ind] = valid_truth_box_index[a]

        # fg truth
        # fg label: for each truth, window with highest overlap, include multiple maxs
        argmax_overlap = np.argmax(overlap,0)
        max_overlap    = overlap[argmax_overlap,np.arange(num_valid_truth_box)]
        argmax_overlap, a = np.where(overlap==max_overlap)
        fg_ind = valid_window_index[argmax_overlap]
        label [fg_ind] = 1
        label_assign[fg_ind] = valid_truth_box_index[a]

        if 0: #<debug>=======================================
            image = input.data.cpu().numpy()*255
            image = image.transpose((1,2,0)).astype(np.uint8).copy()

            images=[]
            for l in range(cfg.rpn_num_heads):
                if l==0:
                    images.append(image.copy())
                else:
                    images.append(cv2.resize(image, None, fx=1/2**l,fy=1/2**l))


            #
            color=plt.get_cmap('cool')(np.arange(0,1,1/(num_truth_box+1))) #'brg'
            color=np.array(color[:,:3])*255
            color=np.fliplr(color)

            for i in range(num_window):
                # if label[i] ==-1 :
                #     x0,y0,x1,y1, l = *window[i], level[i]
                #     cx = int((x1+x0)/2)//2**l
                #     cy = int((y1+y0)/2)//2**l
                #     images[l][cy,cx]=[255,255,255]

                if label[i]>0:
                    c = color[int(assign[i]+1)]
                    x0,y0,x1,y1, l = *window[i], level[i]
                    cx = int((x1+x0)/2)//2**l
                    cy = int((y1+y0)/2)//2**l
                    images[l][cy,cx]=c

            for l in range(cfg.rpn_num_heads):
                if l==0:
                    pass
                else:
                    #print(h)
                    images[l]=cv2.resize(images[l], None, fx=2**l,fy=2**l,interpolation=cv2.INTER_NEAREST)

            images = np.hstack(images)

            image_show('images',images,1)
            image_show('image',image,3)
            cv2.waitKey(0)
            pass
        #<debug>=======================================

        #subsample
        fg_ind = np.where(label==1)[0]
        bg_ind = np.where(label==0)[0]
        fg_ind_length = len(fg_ind)
        bg_ind_length = len(bg_ind)

        #positive labels
        num_fg = int(cfg.rpn_train_fg_fraction * cfg.rpn_train_batch_size)
        if fg_ind_length > num_fg:
            fg_ind = fg_ind[
                np.random.choice( fg_ind_length, size=num_fg, replace=False)
            ]


        # negative labels
        num_bg  = cfg.rpn_train_batch_size - num_fg
        if bg_ind_length > num_bg:
            bg_ind = bg_ind[
                np.random.choice(bg_ind_length, size=num_bg, replace=False)
            ]


        label_weight[fg_ind]=1
        label_weight[bg_ind]=1

        #regression----------------------------------------------
        target_window    = window[fg_ind]
        target_truth_box = truth_box[label_assign[fg_ind]]
        target[fg_ind] = box_transform(target_window, target_truth_box)
        target_weight[fg_ind] = 1


        if 1: #<debug> =========================================================
            image = input.data.cpu().numpy()*255
            image0 = image.transpose((1,2,0)).astype(np.uint8).copy()

            image = image0.copy()
            for b in truth_box:
                x0,y0,x1,y1 =b.astype(np.int32)
                cv2.rectangle(image,(x0,y0),(x1,y1),(255,255,255),1)
                #print(x0,y0,x1,y1, ' : ', x1-x0+1, y1-y0+1)
            image_show('truth_box',image,2)


            image = image0.copy()
            for t in fg_ind:
                x0,y0,x1,y1 = window[t,:4].astype(np.int32)
                cv2.rectangle(image,(x0,y0),(x1,y1),(255,255,255),1)
                #print(x0,y0,x1,y1)
            image_show('target_window',image,2)


            image = image0.copy()
            for t in fg_ind:
                box = box_transform_inv(window[t,:4].reshape(1,4) , target[t].reshape(1,4))
                box = box.reshape(-1)
                x0,y0,x1,y1 = box.astype(np.int32)
                cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,255),1)
                print(target_weight[t], x0,y0,x1,y1)

            print('target_weight.sum(): ',target_weight.sum())
            print('len(fg_ind): ',len(fg_ind))
            image_show('target',image,2)

            print('len(fg_ind): ',len(fg_ind))
            print('len(bg_ind): ',len(bg_ind))
            print('label_weight.sum(): ',label_weight.sum())

            image = image0.copy()
            for t in bg_ind:
                x0,y0,x1,y1 = window[t,:4].astype(np.int32)
                cv2.rectangle(image,(x0,y0),(x1,y1),(255,0,0),1)
                #print(x0,y0,x1,y1)
            image_show('bg_ind',image,2)

            image = image0.copy()
            for t in fg_ind:
                x0,y0,x1,y1 = window[t,:4].astype(np.int32)
                cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),1)
                #print(x0,y0,x1,y1)
            image_show('fg_ind',image,2)


            cv2.waitKey(0)
        #<debug> =========================================================

    # save
    label          = Variable(torch.from_numpy(label)).cuda()
    label_weight   = Variable(torch.from_numpy(label_weight)).cuda()
    target         = Variable(torch.from_numpy(target)).cuda()
    target_weight  = Variable(torch.from_numpy(target_weight)).cuda()
    return  label, label_assign, label_weight, target, target_weight




def make_rpn_target(cfg, inputs, window, truth_boxes, truth_labels):

    rpn_labels = []
    rpn_label_assigns = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input = inputs[b]
        truth_box   = truth_boxes[b]
        truth_label = truth_labels[b]

        rpn_label, rpn_label_assign, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, input, window, truth_box, truth_label)

        rpn_labels.append(rpn_label.view(1,-1))
        #rpn_label_assigns.append(rpn_label_assign.view(1,-1))
        rpn_label_weights.append(rpn_label_weight.view(1,-1))
        rpn_targets.append(rpn_target.view(1,-1,4))
        rpn_targets_weights.append(rpn_targets_weight.view(1,-1))


    rpn_labels          = torch.cat(rpn_labels, 0)
    rpn_label_weights   = torch.cat(rpn_label_weights, 0)
    rpn_targets         = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)

    return rpn_labels, rpn_label_weights, rpn_targets, rpn_targets_weights


## check ############################################################################

def check_layer():
    image_id = '3ebd2ab34ba86e515feb79ffdeb7fc303a074a98ba39949b905dbde3ff4b7ec0'

    dir = '/root/share/project/kaggle/science2018/data/image/stage1_train'
    image_file = dir + '/' + image_id + '/images/' + image_id + '.png'
    npy_file   = dir + '/' + image_id + '/multi_mask.npy'

    multi_mask0 = np.load(npy_file)
    image0      = cv2.imread(image_file,cv2.IMREAD_COLOR)

    batch_size =4
    H,W = 256,256
    images = []
    multi_masks = []
    inputs = []
    boxes  = []
    labels = []
    instances = []
    for b in range(batch_size):
        image, multi_mask = random_crop_transform2(image0, multi_mask0, W, H)
        box, label, instance = multi_mask_to_annotation(multi_mask)
        input = Variable(torch.from_numpy(image.transpose((2,0,1))).float().div(255)).cuda()

        #label[[0,2,3,4,5]]=-1

        images.append(image)
        inputs.append(input)
        multi_masks.append(multi_mask)
        boxes.append(box)
        labels.append(label)
        instances.append(instance)

        # print information ---
        N = len(label)
        for n in range(N):
            print( '%d  :  %s  %d'%(n, box[n], label[n]),)
        print('')

    #dummy features
    in_channels = 256
    num_heads = 4
    feature_heights = [ int(H//2**i) for i in range(num_heads) ]
    feature_widths  = [ int(W//2**i) for i in range(num_heads) ]
    ps = []
    for h,w in zip(feature_heights,feature_widths):
        p = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
        p = Variable(torch.from_numpy(p)).cuda()
        ps.append(p)

    #------------------------

    # check layer
    cfg = type('', (object,), {})() #Configuration() #default configuration
    cfg.rpn_num_heads  = num_heads
    cfg.rpn_num_bases  = 3
    cfg.rpn_base_sizes = [ 8, 16, 32, 64 ] #radius
    cfg.rpn_base_apsect_ratios = [1, 0.5,  2]
    cfg.rpn_strides    = [ 1,  2,  4,  8 ]

    cfg.rpn_train_batch_size     = 256  # rpn target
    cfg.rpn_train_fg_fraction    = 0.5
    cfg.rpn_train_bg_thresh_high = 0.3
    cfg.rpn_train_fg_thresh_low  = 0.7


    #start here --------------------------
    bases, windows = make_rpn_windows(cfg, ps)
    rpn_labels, rpn_label_weights, rpn_targets, rpn_targets_weights = \
        make_rpn_target(cfg, inputs, windows, boxes, labels)

    #
    # for b in range(batch_size):
    #     rpn_label = rpn_labels[b]
    #     rpn_label_weight = rpn_label_weights[b]
    #     rpn_target = rpn_targets[b]
    #     rpn_target_weight = rpn_targets_weights[b]
    #
    #     image = images[b]
    #     label = labels[b]
    #     gt_boxes = label_to_box(label)
    #
    #
    #     image1 = draw_gt_boxes(image, gt_boxes)
    #     #label2d, label_weight2d = draw_rpn_labels2d(rpn_label, rpn_label_weight, feature_widths, feature_heights, cfg.rpn_num_bases )
    #     image2 = draw_rpn_labels(image, windows, rpn_label, rpn_label_weight, is_fg=1, is_bg=1, is_print=1)
    #     image3 = draw_rpn_targets(image, windows, rpn_target, rpn_target_weight, is_before=1, is_after=1, is_print=1)
    #
    #
    #     image_show('image',image,3)
    #     image_show('label',label/label.max()*255,3)
    #
    #     image_show('gt_boxes',image1,3)
    #     # image_show('label2d',label2d,3)
    #     # image_show('label_weight2d',label_weight2d,3)
    #     image_show('image2',image2,3)
    #     image_show('image3',image3,3)
    #
    #
    #
    #     cv2.waitKey(0)



    #im_sh
#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()


 
 