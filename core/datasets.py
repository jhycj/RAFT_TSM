# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
#import torch.nn.functional as F
import torchvision 

import os
#import math
#import random
from glob import glob
import os.path as osp

#from utils import frame_utils
#from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import dataset_config 
from transforms import *


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        
        return int(self._data[2])

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=2, new_length=1, modality='RGB',
                 image_tmpl='{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
     
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                if self.test_mode:             
                    return [Image.open(os.path.join(self.root_path+'/TSM_sthsthv2_imgs/valid', directory, self.image_tmpl.format(idx))).convert('RGB')]
                else: 
                    return [Image.open(os.path.join(self.root_path+'/TSM_sthsthv2_imgs/train', directory, self.image_tmpl.format(idx))).convert('RGB')]

            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(np.random.randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        
        file_name = self.image_tmpl.format(1)
        
        if self.test_mode :
            #self.root_path = self.root_path 
            full_path = os.path.join(self.root_path + '/TSM_sthsthv2_imgs/valid' ,record.path, file_name)
        else: 
            #self.root_path = self.root_path 
            full_path = os.path.join(self.root_path+ '/TSM_sthsthv2_imgs/train', record.path, file_name)
        
        
        while not os.path.exists(full_path):
            print('################## Not Found:', full_path)
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path,record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)
        
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)

        else:

            segment_indices = self._get_test_indices(record)
        
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
   
        import numpy as np 
        x = np.asarray(images[0])  
        process_data = self.transform(images)
        
        # Only for Tiny imiGUE
        if 'imiGUE' in self.root_path :   
            TINYIMIGUE_MAP = {'4': 0, '8': 1, '11':2, '12': 3, '13': 4, '17': 5, 
                              '20': 6, '21': 7, '22': 8, '23': 9, '24': 10, '26': 11, '29': 12}
            
            record.label = str(TINYIMIGUE_MAP[record.label]) 

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


def get_augmentation(args, flip=True):
        
        if args.modality == 'RGB':
            if flip:
                
                return torchvision.transforms.Compose([GroupMultiScaleCrop(args.image_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(args.image_size, [1, .875, .75, .66])])
        elif args.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(args.image_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif args.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(args.image_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])



def fetch_dataloader(args):
    """ Create the data loader for the corresponding training set """

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    train_loader = None 
    val_loader = None 
    
    if args.dataset == 'somethingv2': 

        train_augmentation = get_augmentation(args, flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        scale_size = 224 * 256 // 224
        crop_size = 224
    
        train_dataset = TSNDataSet(
            args.root_path, 
            args.train_list,
            args.num_segments, 
            new_length=1, 
            modality='RGB',
            image_tmpl=prefix, 
            transform= torchvision.transforms.Compose([
                GroupScale(int(scale_size)), 
                train_augmentation,
                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                GroupNormalize(input_mean, input_std)]),
            dense_sample = args.dense_sample)
        
       

        valid_dataset = TSNDataSet(
            args.root_path, 
            args.val_list, 
            args.num_segments,
            new_length=1,
            modality=args.modality,
            image_tmpl=prefix,
            random_shift=False,
            transform=torchvision.transforms.Compose([
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                GroupNormalize(input_mean, input_std)]), 
            dense_sample=args.dense_sample, 
            test_mode= True)
    
    
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            pin_memory= True,
            shuffle=True, 
            drop_last= True  
        )

        val_loader = data.DataLoader(
            valid_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory= True 

        )

        print('Training with %d videos' % len(train_dataset))
    
    return train_loader, val_loader
