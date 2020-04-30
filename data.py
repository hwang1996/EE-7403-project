import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from preprocessing import *


class RetinalDataset(data.Dataset):
    """docstring for RetinalDataset"""
    def __init__(self, phase):
        super(RetinalDataset, self).__init__()


        if phase == 'train':
            train_images = np.zeros((6, 1, 584, 565))
            for i in range(6):
                path = 'data/train/'+str(35+i)+'/'+str(35+i)+'_training.tif'
                train_img = Image.open(path).convert('L')
                train_img = np.array(train_img)
                train_images[i] = train_img
            train_images = pre_processing(train_images)

            gt_images = np.zeros((6, 1, 584, 565))
            for i in range(6):
                path = 'data/gt/'+str(35+i)+'_manual1/'+str(35+i)+'_manual1-0000.jpg'
                gt_img = Image.open(path).convert('1')
                gt_img = np.array(gt_img)
                gt_images[i] = gt_img
            # import pdb; pdb.set_trace()
            # gt_images = gt_images/255.

            train_images = train_images[:,:,9:574,:]
            gt_images = gt_images[:,:,9:574,:]

            self.patches_imgs, self.patches_masks = \
                        extract_random(train_images, gt_images, 48, 48, 190000)

        else:
            # test_img = Image.open('data/train/14_test.tif').convert('L')
            test_img = Image.open('data/train/24_training.tif').convert('L')
            test_input = np.array(test_img)
            test_input = np.expand_dims(test_input, axis=0)
            test_input = np.expand_dims(test_input, axis=0)
            test_input = pre_processing(test_input)
            test_imgs = paint_border(test_input, patch_h=48, patch_w=48)
            self.patches_imgs = extract_ordered(test_imgs, patch_h=48, patch_w=48)


            gt_img = np.array(Image.open('data/gt/24_manual1-0000.jpg').convert('1'))
            # gt_img = gt_img/255.
            gt_img = np.expand_dims(gt_img, axis=0)
            gt_img = np.expand_dims(gt_img, axis=0)
            gt_imgs = paint_border(gt_img, patch_h=48, patch_w=48)
            self.patches_masks = extract_ordered(gt_imgs, patch_h=48, patch_w=48)

    def __getitem__(self, index):

        return self.patches_imgs[index], self.patches_masks[index]

    def __len__(self):
        return self.patches_imgs.shape[0]


