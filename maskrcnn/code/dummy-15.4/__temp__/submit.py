import os, sys
sys.path.append(os.path.dirname(__file__))

from train import *
from metric import *



## overwrite functions ###
def submit_augment(image, index):

    original_image = image.copy()
    #image = resize_to_factor(image, factor=16)
    image = pad_to_factor(image, factor=16)
    input = image.transpose((2,0,1))
    input = torch.from_numpy(input).float().div(255)

    return input, original_image, index


def submit_collate(batch):
    batch_size = len(batch)
    inputs  = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    original_images  =    [batch[b][1]for b in range(batch_size)]
    indices =             [batch[b][2]for b in range(batch_size)]
    return [inputs, original_images, indices]

#--------------------------------------------------------------









def run_submit():

    out_dir  =  RESULTS_DIR + '/mask-rcnn-gray-011b-drop1'
    initial_checkpoint = \
        out_dir +  '/checkpoint/00022000_model.pth'   #00048800_model.pth'
        #out_dir +  '/checkpoint/00042000_model.pth'
        #

    ## setup -----------------------------
    #os.makedirs(csv_dir, exist_ok=True)

    os.makedirs(out_dir +'/submit/overlays', exist_ok=True)
    os.makedirs(out_dir +'/submit/npys', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n')

    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()
    net.load_state_dict(torch.load(initial_checkpoint))
    net.set_mode('eval')

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n\n'%(type(net)))


    ## dataset ---------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
                                'test1_ids_gray_only_53', mode='test',
                                #'train1_ids_gray_only1_500', mode='test',
                                #'valid1_ids_gray_only1_43', mode='test',
                                 transform = submit_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = submit_collate)
    test_num  = len(test_loader.dataset)


    ## start submission here ####################################################################
    start = timer()

    predicts = [];
    n = 0
    for inputs, original_images, indices in test_loader:

        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(n, test_num-1, 100*n/(test_num-1),
                         (timer() - start) / 60), end='',flush=True)
        time.sleep(0.01)

        # forward
        inputs = Variable(inputs,volatile=True).cuda(async=True)
        net( inputs )


        ##save results ---------------------------------------
        batch_size = len(indices)
        assert(batch_size==1)  #note current version support batch_size==1 for variable size input
                               #to use batch_size>1, need to fix code for net.windows, etc

        batch_size,C,H,W = inputs.size()
        ids = test_dataset.ids

        images = inputs.data.cpu().numpy()
        windows          = net.rpn_windows
        rpn_probs_flat   = net.rpn_probs_flat.data.cpu().numpy()
        rpn_deltas_flat  = net.rpn_deltas_flat.data.cpu().numpy()

        proposals  = net.rpn_proposals.data.cpu().numpy()
        detections = net.detections
        masks = net.masks

        for b in range(batch_size):
            original_image = original_images[b]
            height,width = original_image.shape[:2]
            #print(original_image.shape)

            image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
            image = np.clip(image.astype(np.float32)*2,0,255)  #improve contrast
            multi_mask = masks[b]
            #image_show('image_before',image)


            #intermediate results ----
            prob  = rpn_probs_flat[b]
            delta = rpn_deltas_flat[b]
            image_rpn_proposal_before_nms = draw_rpn_proposal_before_nms(image,prob,delta,windows,0.995)

            detection = detections[b]
            image_rcnn_detection_nms = draw_rcnn_detection_nms(image, detection, threshold=0.5)





            #resize to orginal image size ---
            image = image[:height,:width]
            multi_mask = multi_mask[:height,:width]

            #draw results
            color_overlay  = multi_mask_to_color_overlay(multi_mask)
            color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay)
            contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])




            # --------------------------------------------
            image_rpn_proposal_before_nms = image_rpn_proposal_before_nms[:height,:width]
            image_rcnn_detection_nms = image_rcnn_detection_nms[:height,:width]

            all = np.hstack((original_image, contour_overlay, color_overlay,
                             image_rpn_proposal_before_nms, image_rcnn_detection_nms,
                             )).astype(np.uint8)

            id = test_dataset.ids[indices[b]]
            name =id.split('/')[-1]

            #cv2.imwrite(out_dir +'/submit/overlays/%s.multi_mask.png'%(name),multi_mask_overlay)
            #cv2.imwrite(out_dir +'/submit/overlays/%s.contour.png'%(name),contour_overlay)
            cv2.imwrite(out_dir +'/submit/overlays/%s.png'%(name),all)

            np.save(out_dir +'/submit/npys/%s.npy'%(name),multi_mask)
            cv2.imwrite(out_dir +'/submit/npys/%s.png'%(name),color_overlay)




            image_show('all',all)
            #image_show('image',image)
            #image_show('multi_mask_overlay',multi_mask_overlay)
            #image_show('contour_overlay',contour_overlay)
            cv2.waitKey(0)




        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)



#--------------------------------------------------------------

# def do_submit_post_process():
#
#
#     out_dir  = RESULTS_DIR + '/unet-1cls-mask-128-00b'
#     data_dir = out_dir + '/submit'
#
#     ## start -----------------------------
#     os.makedirs(data_dir +'/predict_overlay', exist_ok=True)
#     os.makedirs(data_dir +'/final', exist_ok=True)
#
#
#
#
#     image_files = glob.glob(data_dir + '/predict_mask/*.png')
#     image_files.sort()
#
#     for image_file in image_files:
#         name = image_file.split('/')[-1].replace('.png','')
#
#         image = cv2.imread(DATA_DIR + '/stage1_test/' + name + '/images/' + name +'.png')
#         h,w,  = image.shape[:2]
#
#         mask   = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
#         thresh = (mask>128)
#
#         image_show('image',image)
#         image_show('mask',mask)
#         #cv2.waitKey(0)
#
#
#         #baseline solution -------------------------
#         predict_label = skimage.morphology.label(thresh)
#         predict_label = filter_small(predict_label, threshold=15)
#
#         np.save(data_dir +'/final/' + name + '.npy', predict_label)
#
#
#         #save and show
#         predict_overlay = (skimage.color.label2rgb(predict_label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#
#
#         cv2.imwrite(data_dir +'/predict_overlay/' + name + '.png',predict_overlay)
#         image_show('predict_overlay',predict_overlay)
#         cv2.waitKey(1)


# def do_merge_results():
#
#     data_dir = '/root/share/project/kaggle/science2018/results/unet-1cls-mask-128-00b/submit'
#     image_files = glob.glob(data_dir + '/predict_overlay/*.png')
#     image_files.sort()
#
#     for image_file in image_files:
#         name = image_file.split('/')[-1].replace('.png','')
#         print(name)
#
#         overlay = cv2.imread(image_file)
#
#         #megre
#         image_file1 = '/root/share/project/kaggle/science2018/results/unet-01a/submit/submission-36-fix/' + name + '.png'
#         all = cv2.imread(image_file1)
#         h,w, = all.shape[:2]
#         all[:,w//2:,:]= overlay
#         cv2.imwrite(image_file1,all)
#
#################################################3
## post process ###
def filter_small(multi_mask, threshold):
    num_masks = int(multi_mask.max())

    j=0
    for i in range(num_masks):
        thresh = (multi_mask==(i+1))

        area = thresh.sum()
        if area < threshold:
            multi_mask[thresh]=0
        else:
            multi_mask[thresh]=(j+1)
            j = j+1

    return multi_mask



def run_npy_to_sumbit_csv():

    #npy_dir  = '/root/share/project/kaggle/science2018/results/__submit__/001/npys'
    #csv_file = '/root/share/project/kaggle/science2018/results/__submit__/001/submission.csv'

    npy_dir  = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463/npys'
    csv_file = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463/submission-fix4.csv'

    submit_dir  = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463'
    ## start -----------------------------
    image_dir = '/root/share/project/kaggle/science2018/data/image/stage1_test'
    cvs_ImageId = [];
    cvs_EncodedPixels = [];

    npy_files = glob.glob(npy_dir + '/*.npy')
    for npy_file in npy_files:
        name = npy_file.split('/')[-1].replace('.npy','')

        multi_mask = np.load(npy_file)

        #<todo> ---------------------------------
        # multi_mask1=np.zeros(multi_mask.shape,np.int32)
        # #expand by one
        # num = int( multi_mask.max())
        # for m in range(num):
        #     thresh  =  multi_mask==m+1
        #     contour = thresh_to_inner_contour(thresh)
        #     thresh  = thresh | contour
        #     multi_mask1 [thresh] = m+1
        # multi_mask = multi_mask1


        #post process here
        multi_mask = filter_small(multi_mask, 16)
        #<todo> ---------------------------------

        num = int( multi_mask.max())
        for m in range(num):
            rle = run_length_encode(multi_mask==m+1)
            cvs_ImageId.append(name)
            cvs_EncodedPixels.append(rle)


        #<debug>
        print(num)
        image_file = image_dir +'/' + name + '/images/' + name + '.png'
        image = cv2.imread(image_file)
        color_overlay  = multi_mask_to_color_overlay(multi_mask)
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay)
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])
        image_show('color1_overlay',color1_overlay)
        image_show('contour_overlay',contour_overlay)


        all = np.hstack((image, contour_overlay, color_overlay, color1_overlay,
                         )).astype(np.uint8)
        cv2.imwrite(submit_dir + '/overlays/' + name + '.png', all)

        cv2.waitKey(1)


    #exit(0)
    # submission csv  ----------------------------
    df = pd.DataFrame({ 'ImageId' : cvs_ImageId , 'EncodedPixels' : cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])



def run_sumbit_csv_to_npy():

    submit_dir = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463'
    csv_file = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463/submission.csv'


    ## start -----------------------------
    image_dir = '/root/share/project/kaggle/science2018/data/image/stage1_test'


    os.makedirs(submit_dir + '/npys', exist_ok=True)
    os.makedirs(submit_dir + '/overlays', exist_ok=True)
    os.makedirs(submit_dir + '/psds', exist_ok=True)

    df = pd.read_csv (csv_file)
    names = df['ImageId'].unique()
    #print(len(names))

    for name in names:
        image_file = image_dir + '/' + name +'/images/' + name + '.png'
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        df1 = df.loc[df['ImageId'] == name]
        cvs_EncodedPixels = df1['EncodedPixels'].values
        cvs_EncodedPixels.sort()

        H,W = image.shape[:2]
        multi_mask = np.zeros((H, W), np.int32)
        for t,rle in enumerate(cvs_EncodedPixels):
            thresh = run_length_decode(rle, H, W, fill_value=255)
            multi_mask[thresh>128] = t+1


        color_overlay  = multi_mask_to_color_overlay(multi_mask)
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay)
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])


        all = np.hstack((image, contour_overlay, color_overlay, color1_overlay,
                         )).astype(np.uint8)

        cv2.imwrite(submit_dir + '/overlays/' + name + '.png', all)
        np.save(submit_dir + '/npys/' + name + '.npy', multi_mask)

        #image stack for photoshop  ----------------------------------
        os.makedirs(submit_dir + '/psds/' + name , exist_ok=True)
        cv2.imwrite(submit_dir + '/psds/' + name + '/%s.png'%name, image)
        cv2.imwrite(submit_dir + '/psds/' + name + '/%s.contour.png'%name, contour_overlay)
        cv2.imwrite(submit_dir + '/psds/' + name + '/%s.mask.png'%name, color_overlay)



        #image_show('image',image)
        image_show('all',all)
        cv2.waitKey(1)




def run_png_to_npy():

    png_dir  = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463/add'
    npy_dir  = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.463/add_npys'


    ## start -----------------------------
    image_dir = '/root/share/project/kaggle/science2018/data/image/stage1_test'



    png_files = glob.glob(png_dir + '/*.png')
    for png_file in png_files:
        name = png_file.split('/')[-1].replace('.mask.png','')

        image_file = image_dir +'/' + name + '/images/' + name + '.png'
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        mask_image = cv2.imread(png_file,cv2.IMREAD_COLOR)
        H,W = image.shape[:2]

        image_show('image',image)
        image_show('mask_image',mask_image)

        count=0
        multi_mask = np.zeros((H,W),np.int32)
        unique_color = set( tuple(v) for m in mask_image for v in m )
        print(len(unique_color))
        for color in unique_color:
            print(color)
            if color ==(0,0,0): continue
            if color ==(10,10,10): continue
            thresh = (mask_image==color).all(axis=2)
            label = skimage.morphology.label(thresh)

            index = [label!=0]
            count = multi_mask.max()
            multi_mask[index] =  label[index]+count


            #image_show('thresh',thresh*255)
            #image_show('multi_mask',multi_mask.astype(np.float32)/count*255)

            #cv2.waitKey(0)

        #------------------------------------------------

        #draw results
        color_overlay  = multi_mask_to_color_overlay(multi_mask)
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay)
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])


        np.save(npy_dir +'/%s.npy'%(name),multi_mask)


        print(multi_mask.max())
        all = np.hstack((image, contour_overlay, color_overlay,color1_overlay,
                         )).astype(np.uint8)
        image_show('all',all)

        cv2.waitKey(1)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_submit()
    #run_npy_to_sumbit_csv()
    #do_merge_results()

    #run_png_to_npy()
    run_npy_to_sumbit_csv()

    print('\nsucess!')