from common import *
from dataset.reader import *
from metric import *

from model.mask_rcnn_resnet50_fpn_net import *



def debug_and_draw_one(
        cfg, image, rpn_prob, rpn_delta, proposal, windows, detection, mask):

    image_rpn_proposal_before_nms = draw_rpn_proposal_before_nms(image, rpn_prob, rpn_delta, windows, 0.95)
    image_rpn_proposal_after_nms = draw_rpn_proposal_after_nms(image, proposal, top=100000)
    image_rcnn_detection_nms = draw_rcnn_detection_nms(image, detection, threshold=0.5)
    #print(len(detection))

    multi_mask = mask
    color_overlay   = multi_mask_to_color_overlay(multi_mask)
    contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, color=[0,0,255])

    all=np.hstack((image, image_rpn_proposal_before_nms, image_rpn_proposal_after_nms,
                   image_rcnn_detection_nms, color_overlay, contour_overlay)
            )
    #cv2.imwrite(out_dir +'/train/%05d.png'%indices[b],all)
    #image_show('all',all,1)

    return all



## incomplete implementation: assume only one class
def debug_and_draw_one_detection(cfg, image, detection, truth_box, truth_label):
    truth_box = truth_box[truth_label>0]

    image_detection = image.copy()
    if detection is not None:
        for d in detection:
            x0,y0,x1,y1 = d[1:5].astype(np.int32)
            score = d[5]
            color = to_color(score, [255,0,255])
            cv2.rectangle(image_detection,(x0,y0), (x1,y1), color, 1)

    image_truth = image.copy()
    if truth_box is not None:
        for d in truth_box:
            x0,y0,x1,y1 = d[0:4].astype(np.int32)
            score = 1
            color = to_color(score, [0,255,255])
            cv2.rectangle(image_truth,(x0,y0), (x1,y1), color, 1)
            #print(x0,y0,x1,y1)
    #print()
    # image_show('image_detection',image_detection,4)
    # image_show('image_truth',image_truth,4)
    # cv2.waitKey(0)

    image_detection1 = image.copy()
    image_truth1 = image.copy()
    if truth_box is not None and detection is not None:

        detection = detection[detection[:,5]>cfg.mask_test_nms_threshold]
        if len(detection)!=0:
            box = detection[:,1:5] ##assume one class

            len_box = len(box)
            len_truth_box = len(truth_box)

            overlaps  = cython_box_overlap(box, truth_box)
            argmax_overlaps = np.argmax(overlaps,1)
            max_overlaps = overlaps[np.arange(len_box),argmax_overlaps]

            gt_argmax_overlaps = np.argmax(overlaps,0)
            gt_max_overlaps = overlaps.T[np.arange(len_truth_box),gt_argmax_overlaps]

            miss = np.where(gt_max_overlaps<0.5)[0]
            hit  = np.where(gt_max_overlaps>=0.5)[0]
            fp   = np.where(max_overlaps<0.5)[0]
            tp   = np.where(max_overlaps>=0.5)[0]


            for t in fp:
                x0,y0,x1,y1 = box[t].astype(np.int32)
                cv2.rectangle(image_detection1,(x0,y0), (x1,y1), (255,255,255), 1)
            for t in tp:
                x0,y0,x1,y1 = box[t].astype(np.int32)
                #cv2.rectangle(image_detection1,(x0,y0), (x1,y1), (0,255,255), 1)


            for t in hit:
                x0,y0,x1,y1 = truth_box[t].astype(np.int32)
                #draw_screen_rect(image_truth1,(x0,y0), (x1,y1), (255,255,0), 0.1)

            for t in miss:
                x0,y0,x1,y1 = truth_box[t].astype(np.int32)
                cv2.rectangle(image_truth1,(x0,y0), (x1,y1), (0,0,255), 1)
                #draw_screen_rect(image_truth1,(x0,y0), (x1,y1), (0,0,255), 0.1)


    all = np.hstack([image, image_truth,image_detection,image_truth1,image_detection1])
    #image_show('all detect : truth, detect,  miss,  fp',all,1)

    return all



def debug_and_draw_one_mask(cfg, image, mask, truth_box, truth_label, truth_instance):

    truth_mask = instance_to_multi_mask(truth_instance)
    H,W = image.shape[:2]

    predict = mask!=0
    truth   = np.zeros((H,W),np.bool)
    for t in truth_instance:
         truth = truth | (t>0)

    # image_show('predict',predict*255,1)
    # image_show('truth',truth*255,1)

    hit  = truth & predict
    miss = truth & (~predict)
    fp   = (~truth) & predict

    overlay_results = np.zeros((H,W,3),np.uint8)
    overlay_results[hit ]=[128,128,128]
    overlay_results[miss]=[0,0,255]
    overlay_results[fp  ]=[255,0,0]


    overlay_predict = np.zeros((H,W,3),np.uint8)
    overlay_predict[predict]=[255,0,0]

    overlay_truth = np.zeros((H,W,3),np.uint8)
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


    #iou = num_truth, num_predict
    overlap = np.max(iou,1)
    assign  = np.argmax(iou,1)
    #print(overlap)



    overlay_metric = np.zeros((H,W,3),np.uint8)
    for t in range(num_truth):
        color = to_color(max(0.0,(overlap[t]-0.5)/0.5), [255,255,255])
        overlay_metric[truth_instance[t]>0]=color
    overlay_metric = multi_mask_to_contour_overlay(mask,overlay_metric,[0,0,255])
    draw_shadow_text(overlay_metric,'iou: %0.2f'%overlap.mean(), (5,15),0.5, (255,255,255), 1)


    all = np.hstack((image, overlay_truth, overlay_predict, overlay_results, overlay_metric))
    #image_show('all mask : image, truth, predict, error, metric',all,1)

    return all