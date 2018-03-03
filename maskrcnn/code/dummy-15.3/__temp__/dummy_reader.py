from common import *
from dataset.reader import *


class DummyDataset(Dataset):
    #dummy data for debug
    def __init__(self, split, transform=None, mode='train'):
        super(DummyDataset, self).__init__()
        self.split = split
        self.transform = transform
        self.mode = mode

        ids = ['000','001','002','003','004','005','006','007',]
        self.ids = ids



    def __getitem__(self, index):
        DISK_DIR = '/root/share/project/kaggle/science2018/data/image/disk0'

        id = self.ids[index]
        image = cv2.imread(DISK_DIR + '/' + id + '.png', cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask = image[:,:,0]
            multi_mask = skimage.morphology.label(multi_mask>128)
            meta = 0

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




# check ##################################################################################3
def run_check_dummy_dataset_reader():

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

    #dataset = ScienceDataset(
    dataset = DummyDataset(
        '<not_used>', mode='train',
        transform = augment,
    )
    sampler = SequentialSampler(dataset)
    #sampler = RandomSampler(dataset)


    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        image, multi_mask, box, label, instance, meta, index = dataset[n]

        print(meta)
        image_show('image',image)
        image_show('multi_mask',multi_mask)
        count  = len(instance)
        for i in range(count):
            x0,y0,x1,y1 = box[i]
            cv2.rectangle(instance[i],(x0,y0),(x1,y1),(0,0,255),1)

            image_show('instance[i]',instance[i])
            print('label[i], box[i] : ', label[i], box[i])

            cv2.waitKey(0)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dummy_dataset_reader()

    print( 'sucess!')
