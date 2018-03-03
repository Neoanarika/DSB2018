from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *




def run_png_to_npy():

    png_dir  = '/root/share/project/kaggle/science2018/data/image/disk0'


    ## start -----------------------------
    png_files = sorted(glob.glob(png_dir + '/*.png'))
    for png_file in png_files:
        name = png_file.split('/')[-1].replace('.png','')
        image = cv2.imread(png_file, cv2.IMREAD_COLOR)
        mask_image = image

        H,W = image.shape[:2]

        multi_mask = np.zeros((H,W),np.int32)
        unique_color = set( tuple(v) for m in image for v in m )
        print(len(unique_color))

        count = 0
        for color in unique_color:
            print(color)
            if color ==(0,0,0): continue

            thresh = (mask_image==color).all(axis=2)
            label  = skimage.morphology.label(thresh)

            index = [label!=0]
            multi_mask[index] = label[index]+multi_mask.max()

        #check
        color_overlay   = multi_mask_to_color_overlay  (multi_mask,color='summer')
        color1_overlay  = multi_mask_to_contour_overlay(multi_mask,color_overlay,[255,255,255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask,image,[0,255,0])
        all = np.hstack((image, contour_overlay, color_overlay,color1_overlay,)).astype(np.uint8)

        npy_dir = png_dir + '/' + name
        os.makedirs(npy_dir, exist_ok=True)
        np.save(npy_dir +'/multi_mask.npy',multi_mask)
        cv2.imwrite(npy_dir +'/multi_mask.png',color_overlay)
        cv2.imwrite(npy_dir +'/all.png',all)

        os.makedirs(npy_dir +'/images', exist_ok=True)
        cv2.imwrite(npy_dir +'/images/%s.png'%(name),image)

        image_show('all',all)
        cv2.waitKey(1)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_png_to_npy()

    print( 'sucess!')
