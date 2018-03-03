from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *

from net.lib.box.process import *

#data reader  ----------------------------------------------------------------
MIN_SIZE = 5
IGNORE_BOUNDARY = -1
IGNORE_SMALL    = -2

class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()

        self.split = split
        self.transform = transform
        self.mode = mode

        #read split
        ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')


        #read meta data
        if any(word in split for word in ['train1','valid1','debug1']):
            df = pd.read_csv (DATA_DIR + '/train1_ids_remove_error_668_meta.csv')
            df = df.set_index('name')
            self.df = df
        else:
            self.df = None
            print ('no meta data file')


        #save
        self.ids = ids

        #print
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('\tnum_ids = %d'%(len(self.ids)))
        print('')


    def __getitem__(self, index):
        id = self.ids[index]
        image_id = id.split('/')[-1]
        image = cv2.imread(DATA_DIR + '/image/' + id + '/images/' + image_id +'.png', cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask = np.load( DATA_DIR + '/image/' + id + '/multi_mask.npy').astype(np.int32)

            name = id.split('/')[-1]
            meta = self.df.loc[name].to_dict() if self.df is not None else '<not_used>'

            if self.transform is not None:
                return self.transform(image, multi_mask, meta, index)
            else:
                return input, multi_mask, meta, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)



# draw  ----------------------------------------------------------------
# def multi_mask_to_overlay_0(multi_mask):
#     overlay = skimage.color.label2rgb(multi_mask, bg_label=0, bg_color=(0, 0, 0))*255
#     overlay = overlay.astype(np.uint8)
#     return overlay

def multi_mask_to_color_overlay(multi_mask, image=None, color=None):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_masks))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_masks) ]

    for i in range(num_masks):
        mask = multi_mask==i+1
        overlay[mask]=color[i]
        #overlay = instance[:,:,np.newaxis]*np.array( color[i] ) +  (1-instance[:,:,np.newaxis])*overlay

    return overlay



def multi_mask_to_contour_overlay(multi_mask, image=None, color=[255,255,255]):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    for i in range(num_masks):
        mask = multi_mask==i+1
        contour = mask_to_inner_contour(mask)
        overlay[contour]=color

    return overlay

# modifier  ----------------------------------------------------------------

def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour

def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour

def multi_mask_to_annotation(multi_mask):
    H,W      = multi_mask.shape[:2]
    box      = []
    label    = []
    instance = []

    num_masks = multi_mask.max()
    for i in range(num_masks):
        mask = (multi_mask==(i+1))
        if mask.sum()>1:

            y,x = np.where(mask)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1


            #border = max(2, round(0.2*(w+h)/2))
            border = max(2, round(0.15*min(w,h)))
            #border = 0
            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            #clip
            x0 = max(0,x0)
            y0 = max(0,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)

            #label
            l = 1 #<todo> support multiclass later ... ?
            if is_small_box_at_boundary((x0,y0,x1,y1),W,H,MIN_SIZE):
                l = IGNORE_BOUNDARY
            elif is_small_box((x0,y0,x1,y1),MIN_SIZE):
                l = IGNORE_SMALL

            # add --------------------
            box.append([x0,y0,x1,y1])
            label.append(l)
            instance.append(mask)

    box      = np.array(box,np.float32)
    label    = np.array(label,np.float32)
    instance = np.array(instance,np.float32)

    if len(box)==0:
        box      = np.zeros((0,4),np.float32)
        label    = np.zeros((0,1),np.float32)
        instance = np.zeros((0,H,W),np.float32)

    return box, label, instance


def instance_to_multi_mask(instance):

    H,W = instance.shape[1:3]
    multi_mask = np.zeros((H,W),np.int32)

    num_masks = len(instance)
    for i in range(num_masks):
         multi_mask[instance[i]>0] = i+1

    return multi_mask


# check ##################################################################################3
def run_check_dataset_reader():

    def augment(image, multi_mask, meta, index):
        box, label, instance = multi_mask_to_annotation(multi_mask)

        #for display ----------------------------------------
        #multi_mask = multi_mask/multi_mask.max() *255
        multi_mask = multi_mask_to_color_overlay(multi_mask)

        count  = len(label)
        instance_gray = instance.copy()
        instance =[]
        for i in range(count):
            instance.append(
                cv2.cvtColor((instance_gray[i]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            )
        instance = np.array(instance)
        return image, multi_mask, box, label, instance, meta, index

    dataset = ScienceDataset(
        #'train1_ids_gray_only1_500', mode='train',
        'disk0_ids_dummy_9', mode='train',
        transform = augment,
    )
    #sampler = SequentialSampler(dataset)
    sampler = RandomSampler(dataset)


    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        image, multi_mask, box, label, instance, meta, index = dataset[n]

        print('n=%d------------------------------------------'%n)
        print('meta : ', meta)
        image_show('image',image)
        image_show('multi_mask',multi_mask)

        num_masks  = len(instance)
        for i in range(num_masks):
            x0,y0,x1,y1 = box[i]
            cv2.rectangle(instance[i],(x0,y0),(x1,y1),(0,0,255),1)

            image_show('instance[i]',instance[i])
            print('label[i], box[i] : ', label[i], box[i])

            cv2.waitKey(1)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset_reader()

    print( 'sucess!')
