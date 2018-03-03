import os, sys
sys.path.append(os.path.dirname(__file__))

from train import *
from metric import *
WIDTH, HEIGHT = 128,128


## overwrite functions ###

def eval_augment(image,label,index):

    image,  multi_mask = fix_crop_transform2(image, multi_mask, -1,-1,WIDTH, HEIGHT)

    #---------------------------------------
    box, label, instance  = multi_mask_to_annotation(multi_mask)

    input = image.transpose((2,0,1))
    input = torch.from_numpy(input).float().div(255)

    return input, box, label, instance, index


#eval_augment = valid_augment


#
# def show_evaluate(tensors, labels, probs, indices, ids, wait=1, is_save=True, dir=None):
#
#     os.makedirs(dir, exist_ok=True)
#     os.makedirs(dir + '/all', exist_ok=True)
#     os.makedirs(dir + '/mask', exist_ok=True)
#
#     batch_size,C,H,W = tensors.size()
#     #print(batch_size)
#
#     images = tensors.data.cpu().numpy()
#     labels = labels.data.cpu().numpy()
#     probs  = probs.data.cpu().numpy()
#     for m in range(batch_size):
#         image = images[m].transpose((1,2,0))*255
#         image = image.astype(np.uint8)
#
#         label = labels[m]
#         prob  = probs[m]
#
#         overlay = draw_multi_center(image, prob>0.5)
#
#         label = label.sum(0)
#         prob  = np.clip(prob.sum(0),0,1)
#         label = label.reshape(H,W)[:, :, np.newaxis]*np.array([255,255,255])
#         prob  = prob. reshape(H,W)[:, :, np.newaxis]*np.array([255,255,255])
#         all   = np.hstack((image, overlay, label, prob))
#         all   = all.astype(np.uint8)
#
#         if is_save == True:
#             id = ids[indices[m]]
#             name =id.split('/')[-1]
#             cv2.imwrite(dir +'/all/%s.png'%(name),all)
#             cv2.imwrite(dir +'/mask/%s.png'%(name),prob)
#
#         # image_show('image',image)
#         # image_show('label',label)
#         # image_show('prob',prob)
#         image_show('all',all)
#         cv2.waitKey(wait)





#--------------------------------------------------------------
def run_evaluate():


    out_dir  =  RESULTS_DIR + '/mask-rcnn-gray-011a-debug'
    initial_checkpoint = \
        out_dir +  '/checkpoint/00048800_model.pth'   #00048800_model.pth'
        #

    ## setup  ---------------------------
    os.makedirs(out_dir +'/evaluate/overlays', exist_ok=True)
    os.makedirs(out_dir +'/evaluate/npys', exist_ok=True)
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
                                #'train1_ids_gray_only1_500', mode='train',
                                'valid1_ids_gray_only1_43', mode='train',
                                transform = eval_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)


    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')





    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    net.set_mode('eval')

    test_num  = 0
    test_loss = np.zeros(6,np.float32)
    test_acc  = 0
    for i, (inputs, boxes, labels, instances, indices) in enumerate(test_loader, 0):
        inputs = Variable(inputs).cuda()

        net(inputs, boxes,  labels, instances )
        loss = net.loss(inputs, boxes,  labels, instances)



        ##save results ---------------------------------------
        batch_size = len(indices)
        assert(batch_size==1)  #note current version support batch_size==1 for variable size input
                               #to use batch_size>1, need to fix code for net.windows, etc

        batch_size,C,H,W = inputs.size()
        ids = test_dataset.ids

        images = inputs.data.cpu().numpy()
        masks = net.masks

        for b in range(batch_size):
            original_image = original_images[b]
            height,width = original_image.shape[:2]
            #print(original_image.shape)

            image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
            image = np.clip(image.astype(np.float32)*2.5,0,255)  #improve contrast
            multi_mask = masks[b]
            #image_show('image_before',image)

            #resize to orginal image size,
            image = image[:height,:width]
            multi_mask = multi_mask[:height,:width]

            #draw results
            multi_mask_overlay = multi_mask_to_overlay(multi_mask)

            contour_overlay = image.copy()
            contour_overlay[np.where((multi_mask_overlay==[255,255,255]).all(axis=2))] = [0,255,0]


            # --------------------------------------------
            all = np.hstack((original_image, contour_overlay, multi_mask_overlay, )).astype(np.uint8)

            id = test_dataset.ids[indices[b]]
            name =id.split('/')[-1]

            #cv2.imwrite(out_dir +'/submit/overlays/%s.multi_mask.png'%(name),multi_mask_overlay)
            #cv2.imwrite(out_dir +'/submit/overlays/%s.contour.png'%(name),contour_overlay)
            cv2.imwrite(out_dir +'/evaluate/overlays/%s.png'%(name),all)

            np.save(out_dir +'/evaluate/npys/%s.npy'%(name),multi_mask)
            cv2.imwrite(out_dir +'/evaluate/npys/%s.png'%(name),multi_mask_overlay)




            image_show('all',all)
            #image_show('image',image)
            #image_show('multi_mask_overlay',multi_mask_overlay)
            #image_show('contour_overlay',contour_overlay)
            cv2.waitKey(0)

        # print statistics  ------------
        test_acc  += 0 #batch_size*acc[0][0]
        test_loss += batch_size*np.array((
                           loss .cpu().data.numpy()[0],
                           net.rpn_cls_loss.cpu().data.numpy()[0],
                           net.rpn_reg_loss.cpu().data.numpy()[0],
                           net.rcnn_cls_loss.cpu().data.numpy()[0],
                           net.rcnn_reg_loss.cpu().data.numpy()[0],
                           net.mask_cls_loss.cpu().data.numpy()[0],
                         ))
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_acc  = %0.5f\n'%(test_acc))
    log.write('test_loss = %0.5f\n'%(test_loss))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')



## post process here ####-------------------------------------
def run_evaluate_map():

    out_dir = RESULTS_DIR + '/mask-rcnn-gray-011b-drop1'
    split   = 'valid1_ids_gray_only_43'

    #------------------------------------------------------------------
    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')

    #os.makedirs(out_dir +'/eval/'+split+'/label', exist_ok=True)
    #os.makedirs(out_dir +'/eval/'+split+'/final', exist_ok=True)


    image_files = glob.glob(out_dir + '/submit/npys/*.png')
    image_files.sort()

    average_precisions = []
    for image_file in image_files:
        #image_file = image_dir + '/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'

        name  = image_file.split('/')[-1].replace('.png','')

        image   = cv2.imread(DATA_DIR + '/image/stage1_train/' + name + '/images/' + name +'.png')
        truth   = np.load(DATA_DIR    + '/image/stage1_train/' + name + '/multi_mask.npy').astype(np.int32)
        predict = np.load(out_dir     + '/submit/npys/' + name + '.npy').astype(np.int32)
        assert(predict.shape == truth.shape)
        assert(predict.shape[:2] == image.shape[:2])


        #image_show('image',image)
        #image_show('mask',mask)
        #cv2.waitKey(0)


        #baseline labeling  -------------------------


        # fill hole, file small, etc ...
        # label = filter_small(label, threshold=15)


        average_precision, precision = compute_average_precision(predict, truth)
        average_precisions.append(average_precision)

        #save and show  -------------------------
        print(average_precision)

        # overlay = (skimage.color.label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
        # cv2.imwrite(out_dir +'/eval/'+split+'/label/' + name + '.png',overlay)
        # np.save    (out_dir +'/eval/'+split+'/label/' + name + '.npy',label)


        # overlay1 = draw_label_contour (image, label )
        # mask  = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        # final = np.hstack((image, overlay1, overlay, mask))
        # final = final.astype(np.uint8)
        # cv2.imwrite(out_dir +'/eval/'+split+'/final/' + name + '.png',final)
        #
        #
        # image_show('image',image)
        # image_show('mask',mask)
        # image_show('overlay',overlay)
        # cv2.waitKey(1)

    ##----------------------------------------------
    average_precisions = np.array(average_precisions)
    log.write('-------------\n')
    log.write('average_precision = %0.5f\n'%average_precisions.mean())
    log.write('\n')









# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_evaluate()
    run_evaluate_map()


    print('\nsucess!')