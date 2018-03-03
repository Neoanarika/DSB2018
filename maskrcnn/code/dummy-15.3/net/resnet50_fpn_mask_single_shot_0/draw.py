from common import *
from metric import *
from dataset.reader import *

from net.resnet50_fpn_mask_single_shot_0.model import *




def draw_multi_rpn_prob(cfg, image, rpn_prob_flat):

    H,W = image.shape[:2]
    num_scales = len(cfg.rpn_scales)
    num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]

    rpn_prob = (rpn_prob_flat[:,1]*255).astype(np.uint8)
    rpn_prob = unflat_to_c3(rpn_prob, num_scales, num_bases, H, W)

    ## -pyramid -
    pyramid=[]
    for l in range(num_scales):
        if l==0:
            pyramid.append(image.copy())
        else:
            pyramid.append(cv2.resize(image, None, fx=1/2**l,fy=1/2**l))

    all = []
    for l in range(num_scales):
        a = np.vstack((
            pyramid[l],
            rpn_prob[l],
        ))

        all.append(
            cv2.resize(a, None, fx=2**l,fy=2**l,interpolation=cv2.INTER_NEAREST)
        )

    all = np.hstack(all)
    return all



def draw_multi_rpn_delta(cfg, image, rpn_prob_flat, rpn_delta_flat, window):

    threshold = cfg.rpn_test_nms_pre_score_threshold

    image = image.copy()
    index = np.where(rpn_prob_flat>threshold)[0]
    for i in index:
        l = np.argmax(rpn_prob_flat[i])
        if l==0: continue

        w = window[i]
        t = rpn_delta_flat[i,l]
        b = box_transform_inv(w.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        #cv2.rectangle(image,(w[0], w[1]), (w[2], w[3]), (255,255,255), 1)
        cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), (0,0,255), 1)

    return image



def draw_multi_rpn_proposal(cfg, image, proposal):

    image = image.copy()
    for p in proposal:
        x0,y0,x1,y1 = p[1:5].astype(np.int32)
        score = p[5]
        color = to_color(score, [255,0,255])
        cv2.rectangle(image, (x0,y0), (x1,y1), color, 1)

    return image




def draw_truth_box(cfg, image, truth_box, truth_label):

    image = image.copy()
    if len(truth_box)>0:
        for b,l in zip(truth_box, truth_label):
            x0,y0,x1,y1 = b.astype(np.int32)
            if l <=0 : continue

            cv2.rectangle(image, (x0,y0), (x1,y1), [0,255,0], 1)

    return image


def draw_multi_rpn_metric(cfg, image, proposal, truth_box, truth_label):

    image_truth    = image.copy()
    image_proposal = image.copy()
    image_hit      = image.copy()
    image_miss     = image.copy()
    image_fp       = image.copy()
    image_invalid  = image.copy()
    precision = 0

    if len(proposal)>0 and len(truth_box)>0:

        thresholds=[0.5,0.7]

        box = proposal[:,1:5]
        precisions, recalls, results, truth_results = \
            compute_precision_for_box(box, truth_box, truth_label, thresholds)

        #for precision, recall, result, truth_result, threshold in zip(precisions, recalls, results, truth_results, thresholds):

        if 1:
            precision, recall, result, truth_result, threshold = \
                precisions[0], recalls[0], results[0], truth_results[0], thresholds[0]


            for i,b in enumerate(truth_box):
                x0,y0,x1,y1 = b.astype(np.int32)

                if truth_result[i]==HIT:
                    cv2.rectangle(image_truth,(x0,y0), (x1,y1), (0,255,255), 1)
                    draw_screen_rect(image_hit,(x0,y0), (x1,y1), (255,255,0), 0.25)

                if truth_result[i]==MISS:
                    cv2.rectangle(image_truth,(x0,y0), (x1,y1), (0,255,255), 1)
                    cv2.rectangle(image_miss,(x0,y0), (x1,y1), (0,0,255), 1)

                if truth_result[i]==INVALID:
                    draw_screen_rect(image_invalid,(x0,y0), (x1,y1), (255,255,255), 0.5)

            for i,b in enumerate(box):
                x0,y0,x1,y1 = b.astype(np.int32)
                cv2.rectangle(image_proposal,(x0,y0), (x1,y1), (255,0,255), 1)

                if result[i]==TP:
                    cv2.rectangle(image_hit,(x0,y0), (x1,y1), (255,255,0), 1)

                if result[i]==FP:
                    cv2.rectangle(image_fp,(x0,y0), (x1,y1), (0,255,0), 1)

                if result[i]==INVALID:
                    cv2.rectangle(image_invalid,(x0,y0), (x1,y1), (255,255,255), 1)

    draw_shadow_text(image_hit, 'hit',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_miss,'miss', (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_fp,  'fp',   (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_invalid, 'n.a.', (5,15),0.5, (255,255,255), 1)

    all = np.hstack([image_truth,image_proposal,image_hit,image_miss,image_fp,image_invalid])
    draw_shadow_text(all,'%0.2f prec@0.5'%precision, (5,15),0.5, (255,255,255), 1)
    return all


def draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance):


    H,W = image.shape[:2]
    overlay_truth   = np.zeros((H,W,3),np.uint8)
    overlay_predict = np.zeros((H,W,3),np.uint8)
    overlay_error   = np.zeros((H,W,3),np.uint8)
    overlay_metric  = np.zeros((H,W,3),np.uint8)
    average_overlap   = 0
    average_precision = 0
    precision_50 = 0
    precision_70 = 0

    if len(truth_box)>0:

        #pixel error: fp and miss
        truth_mask = instance_to_multi_mask(truth_instance)
        truth   = np.zeros((H,W),np.bool)
        for t in truth_instance:
             truth = truth | (t>0)

        predict = mask!=0

        hit  = truth & predict
        miss = truth & (~predict)
        fp   = (~truth) & predict

        overlay_error[hit ]=[128,128,128]
        overlay_error[miss]=[0,0,255]
        overlay_error[fp  ]=[255,0,0]

        # truth and predict
        overlay_predict[predict]=[255,0,0]
        overlay_truth[truth]=[0,0,255]
        overlay_predict = multi_mask_to_contour_overlay(mask,overlay_predict,[255,255,255])
        overlay_truth   = multi_mask_to_contour_overlay(truth_mask,overlay_truth,[255,255,255])


        # metric -----
        predict = mask
        truth   = np.zeros((H,W),np.int32)
        for l,t in enumerate(truth_instance):
            truth[t>0] = l+1

        num_truth   = len(np.unique(truth  ))-1
        num_predict = len(np.unique(predict))-1

        if num_predict!=0:
            intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth+1, num_predict+1))[0]

            # Compute areas (needed for finding the union between all objects)
            area_true = np.histogram(truth,   bins = num_truth  +1)[0]
            area_pred = np.histogram(predict, bins = num_predict+1)[0]
            area_true = np.expand_dims(area_true, -1)
            area_pred = np.expand_dims(area_pred,  0)
            union = area_true + area_pred - intersection
            intersection = intersection[1:,1:]   # Exclude background from the analysis
            union = union[1:,1:]
            union[union == 0] = 1e-9
            iou = intersection / union # Compute the intersection over union

            precision = {}
            average_precision = 0
            thresholds = np.arange(0.5, 1.0, 0.05)
            for t in thresholds:
                tp, fp, fn = compute_precision(t, iou)
                prec = tp / (tp + fp + fn)
                precision[round(t,2) ]=prec
                average_precision += prec
            average_precision /= len(thresholds)
            precision_50 = precision[0.50]
            precision_70 = precision[0.70]


            #iou = num_truth, num_predict
            overlap = np.max(iou,1)
            assign  = np.argmax(iou,1)
            #print(overlap)

            for t in range(num_truth):
                color = to_color(max(0.0,(overlap[t]-0.5)/0.5), [0,255,255])
                overlay_metric[truth_instance[t]>0]=color
            overlay_metric = multi_mask_to_contour_overlay(mask,overlay_metric,[255,0,0])
            average_overlap = overlap.mean()

    draw_shadow_text(overlay_truth,   'truth',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_predict, 'predict',(5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_error,   'error',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_metric,  '%0.2f iou '%average_overlap,    (5,15),0.5, (255,255,255), 1)

    all = np.hstack((image, overlay_truth, overlay_predict, overlay_error, overlay_metric))
    draw_shadow_text(all,'%0.2f prec@0.5'%(precision_50), (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(all,'%0.2f prec@0.7'%(precision_70), (5,30),0.5, (255,255,255), 1)
    draw_shadow_text(all,'%0.2f prec'%average_precision,  (5,H-15),0.5, (0,255,255), 1)
    #image_show('all mask : image, truth, predict, error, metric',all,1)

    return all