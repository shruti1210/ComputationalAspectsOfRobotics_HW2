import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            dataset_dir: string, train_dir, val_dir, and test_dir in segmentation.py.
                         Be careful the images are stored in the subfolders under these directories.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        # TODO: transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_rgb, std_rgb),])
        
        # TODO: number of samples in the dataset.
        #  You'd better not hard code the number,
        #  because this class is used to create train, validation and test dataset (which have different sizes).
        
        self.dataset_length = len(os.listdir(self.dataset_dir  + 'rgb/'))
        
        #self.dataset_length = len(os.listdir(os.path.join(dataset_dir, 'rgb')))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        sample = {}
        if self.has_gt is False:
            #rgb_img = image.read_rgb(os.path.join(self.dataset_dir, 'rgb', f'{idx}_rgb.png'))
            rgb_img = image.read_rgb(self.dataset_dir + 'rgb/' + f"{idx}_rgb.png")
            rgb_img = self.transform(rgb_img) 
            sample = {'input': rgb_img}
        else:
            #rgb_img = image.read_rgb(os.path.join(self.dataset_dir, 'rgb', f'{idx}_rgb.png'))
            rgb_img = image.read_rgb(self.dataset_dir + 'rgb/' + f"{idx}_rgb.png")
            rgb_img = self.transform(rgb_img) 
            gt_mask = torch.LongTensor(image.read_mask(self.dataset_dir + 'gt/' + f"{idx}_gt.png"))
            #gt_mask = torch.LongTensor(image.read_mask(os.path.join(self.dataset_dir, 'gt', f'{idx}_gt.png')))
            sample = {'input': rgb_img, 'target': gt_mask}
        return sample
