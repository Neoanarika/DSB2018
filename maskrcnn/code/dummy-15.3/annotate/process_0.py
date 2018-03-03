from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *



def instance_to_box(instance):
    H,W = instance.shape[:2]

    y,x = np.where(instance>0)
    y0 = y.min()
    y1 = y.max()
    x0 = x.min()
    x1 = x.max()
    w = (x1-x0)+1
    h = (y1-y0)+1

    #border = max(2, round(0.2*(w+h)/2))
    border = max(2, round(0.1666667*min(w,h)))

    #border = 0
    x0 = x0-border
    x1 = x1+border
    y0 = y0-border
    y1 = y1+border

    #clip
    x0 = int(max(0,x0))
    y0 = int(max(0,y0))
    x1 = int(min(W-1,x1))
    y1 = int(min(H-1,y1))

    return x0,y0,x1,y1



#extra processing
def run_process_box():

    split = 'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')


    #lower,upper = 0,12
    lower,upper = 10,26

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        name = id.split('/')[-1]

        image_files = glob.glob(DATA_DIR + '/image/' + id + '/images/*.png')
        assert(len(image_files)==1)
        image_file=image_files[0]
        #print(id)

        #----start -----------------------------
        num_valid = 0
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        H,W,C = image.shape
        instance_files = sorted(glob.glob(DATA_DIR + '/image/' + id + '/masks/*.png'))

        num = len(instance_files)
        for i in range(num):
            instance_file = instance_files[i]
            instance = cv2.imread(instance_file,cv2.IMREAD_GRAYSCALE)
            instance = instance >128

            x0,y0,x1,y1 = instance_to_box(instance)
            cv2.rectangle(image, (x0,y0),(x1,y1),(0,0,255),1)

            w=x1-x0
            h=y1-y0
            size =(w*h)**0.5
            valid = (lower<size) & (size<upper)
            num_valid += valid

            if valid:
                cv2.rectangle(image, (x0,y0),(x1,y1),(255,0,255),2)

        ratio = num_valid/num
        #if ratio>=0.75 and num>=3 :
        if 1:
            print('%s   %5d / %5d  %0.3f'%(id,num_valid,num,ratio))
            image_show('image',image)
            cv2.waitKey(0)

    zz=0

##------------------------------------------------------------------------------------------------


#extra processing
def run_process_box0():

    split = 'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')


    boxes=[]
    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        name = id.split('/')[-1]

        image_files = glob.glob(DATA_DIR + '/image/' + id + '/images/*.png')
        assert(len(image_files)==1)
        image_file=image_files[0]
        print(id)

        #----start -----------------------------

        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        H,W,C = image.shape
        instance_files = sorted(glob.glob(DATA_DIR + '/image/' + id + '/masks/*.png'))
        count = len(instance_files)
        for i in range(count):
            instance_file = instance_files[i]
            instance = cv2.imread(instance_file,cv2.IMREAD_GRAYSCALE)
            instance = instance >128

            x0,y0,x1,y1 = instance_to_box(instance)
            cv2.rectangle(image, (x0,y0),(x1,y1),(0,0,255),1)

            boxes.append([x0,y0,x1,y1])

        image_show('image',image)
        cv2.waitKey(1)

    zz=0










def run_delete_annotation():

    split = 'train1_ids_all_670'
    ids   = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        print(id)

        #----clear old -----------------------------
        if 1:
            files = glob.glob(DATA_DIR + '/image/' + id + '/*.png')
            files = [os.path.basename(f) for f in files]

            for f in files:
                file = DATA_DIR + '/image/' + id + '/' + f
                if os.path.exists(file):
                    os.remove(file)
        #----clear old -----------------------------





#extra processing
def run_make_annotation():

    split = 'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        image_files =   glob.glob(DATA_DIR + '/image/' + id + '/images/*.png')
        assert(len(image_files)==1)
        image_file=image_files[0]
        print(id)

        #----clear old -----------------------------
        if 1:
            for f in ['one_mask.png','one_countour_mask.png','one_countour_image.png','one_countour.png',
                      'overlap.png', 'one_center.png','/masks.npy', '/labels.npy',
                      '/countour_on_image.png', '/cut_mask.png', '/label.npy', '/mask.png','/overlay.png',
                      '/multi.npy','/multi.png',
                      '/instance.npy','/instance.png',
                      '/multi_instance.npy','/multi_instance.png',
                      ]:
                file = DATA_DIR + '/image/' + id + '/' + f
                if os.path.exists(file):
                    os.remove(file)
        #----clear old -----------------------------


        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        H,W,C = image.shape
        multi_mask = np.zeros((H,W), np.int32)
        mask     = np.zeros((H,W), np.uint8)
        countour = np.zeros((H,W), np.uint8)




        mask_files = glob.glob(DATA_DIR + '/image/' + id + '/masks/*.png')
        mask_files.sort()
        count = len(mask_files)
        for i in range(count):
            mask_file = mask_files[i]
            thresh = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            thresh = thresh >128
            index  = np.where(thresh==True)

            multi_mask[thresh] = i+1
            mask  = np.logical_or(mask,thresh)
            countour = np.logical_or(countour, thresh_to_inner_contour(thresh) )



        ## save and show -------------------------------------------
        countour_on_image = image.copy()
        countour_on_image = countour[:,:,np.newaxis]*np.array((0,255,0)) +  (1-countour[:,:,np.newaxis])*countour_on_image

        countour_overlay  = countour*255
        mask_overlay  = mask*255
        multi_mask_overlay = multi_mask_to_overlay(multi_mask)


        image_show('image',image)
        image_show('mask', mask_overlay)
        image_show('multi_mask',multi_mask_overlay)
        image_show('countour',countour_overlay)
        image_show('countour_on_image',countour_on_image)



        np.save(DATA_DIR + '/image/' + id + '/multi_mask.npy', multi_mask)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/multi_mask.png',multi_mask_overlay)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/mask.png',mask_overlay)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/countour.png',countour_overlay)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/countour_on_image.png',countour_on_image)

        cv2.waitKey(1)









#extra processing
def run_make_psd():

    #output
    psd_dir = '/root/share/project/kaggle/science2018/data/others/psd/stage1_train'

    #input
    split = 'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        name = id.split('/')[-1]

        image_files = glob.glob(DATA_DIR + '/image/' + id + '/images/*.png')
        assert(len(image_files)==1)
        image_file=image_files[0]
        print(id)

        #-----------------------------------------------------------
        #image
        image      = cv2.imread(image_file,cv2.IMREAD_COLOR)
        norm_image = np.clip(image.astype(np.float32)+96, 0, 255)  #improve contrast
        #norm_image = np.clip(image.astype(np.float32)*2, 0, 255)  #improve contrast

        #multi mask
        H,W = image.shape[:2]
        multi_mask = np.zeros((H,W), np.int32)
        mask_files = sorted( glob.glob(DATA_DIR + '/image/' + id + '/masks/*.png'))
        count = len(mask_files)

        for i in range(count):
            mask_file = mask_files[i]
            instance = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            instance = instance >128
            multi_mask[instance]= i+1


        color_overlay   = multi_mask_to_color_overlay(multi_mask,color='summer')
        color_overlay1  = multi_mask_to_contour_overlay(multi_mask, color_overlay,[255,255,255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, norm_image,   [0,255,0]    )

        all = np.hstack((image,contour_overlay,color_overlay1,color_overlay))

        ## save and show
        image_show('all',all)

        dir = psd_dir +'/%s'%name
        os.makedirs(dir,exist_ok=True)
        cv2.imwrite(dir + '/%s.png'%name, image)
        cv2.imwrite(dir + '/%s.contour.png'%name,contour_overlay)
        cv2.imwrite(dir + '/%s.mask.png'%name,color_overlay)
        cv2.imwrite(psd_dir + '/%s.all.png'%name, all)

        cv2.waitKey(1)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_make_psd()
    run_process_box()

    print( 'sucess!')
