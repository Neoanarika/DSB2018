#  caffe-fast-rcnn/src/caffe/layers/smooth_L1_loss_layer.cu
#
#  sigma normlisation:
#     https://github.com/rbgirshick/py-faster-rcnn
#        see smooth_l1_loss_param { sigma: 3.0 }
#
#  std normlisation:
#        see cfg.TRAIN.BBOX_NORMALIZE_STDS



##-----------------------------------------------------------------
'''
http://pytorch.org/docs/0.1.12/_modules/torch/nn/modules/loss.html
Huber loss

class SmoothL1Loss(_Loss):
                          { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
    loss(x, y) = 1/n \sum {
                          { |x_i - y_i| - 0.5,   otherwise

    # loss = diff/(no._of_samples * dim_of_one_sample)
'''
#debug (cross check)
#l = modified_smooth_l1(rpn_deltas, rpn_targets, deltas_sigma)
##-----------------------------------------------------------------


from common import *

# focal loss
# https://github.com/andreaazzini/retinanet.pytorch/blob/master/loss.py
# https://github.com/unsky/focal-loss
# https://github.com/zimenglan-sysu-512/paper-note/blob/master/focal_loss.pdf
# https://github.com/pytorch/pytorch/issues/563
def weighted_focal_loss_for_cross_entropy(logits, labels, weights, gamma=2.):
    # cross_entropy_loss = - log(p[t])
    # focal_loss         = -(1-p[t])**gamma * log(p[t])

    # y = torch.FloatTensor(batch_size, num_classes).zero_().cuda()
    # y.scatter_(1, labels, 1)


    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)

    loss = - weights * log_probs * (1 - probs) ** gamma
    loss = loss.sum()/(weights.sum()+1e-12)

    return loss.sum()


##  http://geek.csdn.net/news/detail/126833
def weighted_binary_cross_entropy_with_logits(logits, labels, weights):

    loss = weights*(logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs())))
    loss = loss.sum()/(weights.sum()+1e-12)

    return loss


def weighted_cross_entropy_with_logits(logits, labels, weights):

    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    loss = - weights * log_probs
    loss = loss.sum()/(weights.sum()+1e-12)

    return loss


# original F1 smooth loss from rcnn
def weighted_smooth_l1( predicts, targets, weights, sigma=3.0):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise

        inside_weights  = 1
        outside_weights = 1/num_examples
    '''

    predicts = predicts.view(-1)
    targets  = targets.view(-1)
    weights  = weights.view(-1)

    sigma2 = sigma * sigma
    diffs  =  predicts-targets
    smooth_l1_signs = torch.abs(diffs) <  (1.0 / sigma2)
    smooth_l1_signs = smooth_l1_signs.type(torch.cuda.FloatTensor)

    smooth_l1_option1 = 0.5 * diffs* diffs *  sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5  / sigma2
    loss = weights*(smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs))

    loss = loss.sum()/(weights.sum()+1e-12)

    return loss


#---------------------------------------------------------------------------
#<todo> support multi-class
# default loss
# def rpn_loss(logits, deltas, labels, label_weights,
#              targets, target_weights,  deltas_sigma=3.0):
#
#     batch_size, num_windows = logits.size()
#     target_weights = target_weights.view((batch_size, num_windows, 1)).expand((batch_size, num_windows, 4)).contiguous()
#
#     rpn_cls_loss  = weighted_binary_cross_entropy_with_logits(logits, labels, label_weights)
#     rpn_reg_loss  = weighted_smooth_l1( deltas, targets, target_weights, deltas_sigma)
#
#     return rpn_cls_loss, rpn_reg_loss

#<todo> support multi-class
# using focal loss
def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights,  delta_sigma=3.0):

    batch_size, num_windows, num_classes = logits.size()
    labels = labels.long()

    #classification ---
    batch_size    = batch_size*num_windows
    logits        = logits.view(batch_size, num_classes)
    labels        = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)
    rpn_cls_loss  = weighted_focal_loss_for_cross_entropy(logits, labels, label_weights)

    #regression ---
    #
    # # one hot encode
    deltas          = deltas.view  (batch_size, num_classes,4)
    targets         = targets.view (batch_size, 4)
    target_weights  = target_weights.view (batch_size, 1)
    index = (labels!=0).nonzero()[:,0]

    deltas  = deltas [index]
    targets = targets[index]
    target_weights = target_weights[index].expand((-1,4)).contiguous()

    select = labels[index].view(-1,1).expand((-1,4)).contiguous().view(-1,1,4)
    deltas = deltas.gather(1,select)

    rpn_reg_loss = weighted_smooth_l1( deltas, targets, target_weights, delta_sigma)
    #rpn_reg_loss = Variable(torch.zeros((1))).cuda()


    return rpn_cls_loss, rpn_reg_loss


#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #check_layer()

 
 