import os
import gc
import cv2
import random
import pickle

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from utils import calculate_iou, convert_from_corners_to_midpoint
from pascal_voc import VOC2007, VOC2012

def selective_search(image):
    # return region proposals of selective searh over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()


class RCNNDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, dataset, cfg,
                 IoU_threshold={'positive': 0.5, 'partial': 0.3},
                 sample_ratio=(32, 96),
                 padding=16,
                 data_path='../'):

        self.data_path = data_path
        self.dataset = dataset
        self.padding = padding
        self.IoU_threshold = IoU_threshold
        self.sample_ratio = sample_ratio
        self.image_dir = self.dataset.train_dir + '/JPEGImages/'
        self.annot_dir = self.dataset.train_dir + '/Annotations/'

        self.transform = transforms.Compose([  # preprocess image
            transforms.Resize((cfg['image_size'], cfg['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if not os.path.exists(self.dataset.train_dir + '/SelectiveSearchImages/'):
            os.mkdir(self.dataset.train_dir + '/SelectiveSearchImages/')
            os.mkdir(self.dataset.train_dir + '/SelectiveSearchLabels/')
        self._generate_dataset(sample_ratio, IoU_threshold, padding)

        self.train_images = os.listdir(self.dataset.train_dir + '/SelectiveSearchImages/')
        self.train_labels = os.listdir(self.dataset.train_dir + '/SelectiveSearchLabels/')

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.dataset.train_dir + '/SelectiveSearchImages/' + self.train_images[idx])
        with open(self.dataset.train_dir + '/SelectiveSearchLabels/' + self.train_labels[idx], 'rb') as f:
            label = pickle.load(f)

        return {'image': self.transform(image),
                'label': label[0],
                'est_bbox': label[1],
                'gt_bbox': label[2]}

    def _generate_dataset(self, sample_ratio, IoU_threshold, padding=16):
        img_list = os.listdir(self.image_dir)
        annot_list = os.listdir(self.annot_dir)

        obj_counter = 0
        bg_counter = 0
        self.train_images = []
        self.train_labels = []

        print('[*] Generating dataset for R-CNN.')
        pbar = tqdm(zip(img_list, annot_list), position=0, leave=True)  # only 500 images

        start = False
        for img_name, annot_name in pbar:
            if '2011_006295' in img_name:
                start = True

            if not start:
                continue

            pbar.set_description(f"Data size: {len(self.train_labels)}")
            gc.collect()

            # load image & gt bounding boxes
            image = cv2.imread(self.image_dir + img_name)
            xml_path = self.annot_dir + annot_name

            try:
                gt_bboxes = self.dataset.read_xml(xml_path)
            except:
                continue

            # generete bbox proposals via selective search
            rects = selective_search(image)[:2000]  # parse first 2000 boxes
            random.shuffle(rects)

            file_counter = 0
            # loop through all ss bounding box proposals
            for (x, y, w, h) in rects:
                # apply padding
                x1, x2 = np.clip([x - padding, x + w + padding], 0, image.shape[1])
                y1, y2 = np.clip([y - padding, y + h + padding], 0, image.shape[0])
                bbox_est = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

                # check the proposal with every elements of the gt boxes
                is_object = False  # define flag
                for gt_bbox in gt_bboxes:
                    iou = calculate_iou(gt_bbox, bbox_est, box_format='corners')

                    if iou >= IoU_threshold['positive']:  # if object(RoI > 0.5)
                        obj_counter += 1
                        cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]

                        est_bbox_xywh = convert_from_corners_to_midpoint(bbox_est)
                        gt_bbox_xywh = convert_from_corners_to_midpoint(gt_bbox)

                        cv2.imwrite(self.dataset.train_dir + f'/SelectiveSearchImages/{img_name[:-4]}_{file_counter}.jpg', cropped)
                        with open(self.dataset.train_dir + f'/SelectiveSearchLabels/{img_name[:-4]}_{file_counter}.pkl', 'wb') as file:
                            pickle.dump([gt_bbox['class'], est_bbox_xywh, gt_bbox_xywh], file)
                            
                        file_counter += 1
                        is_object = True
                        break

                # if the object is not close to any g.t bbox
                if bg_counter < sample_ratio[1] and not is_object:
                    bg_counter += 1
                    cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]

                    est_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                    gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)

                    cv2.imwrite(self.dataset.train_dir + f'/SelectiveSearchImages/{img_name[:-4]}_{file_counter}.jpg', cropped)
                    with open(self.dataset.train_dir + f'/SelectiveSearchLabels/{img_name[:-4]}_{file_counter}.pkl', 'wb') as file:
                        pickle.dump([0, est_bbox_xywh, gt_bbox_xywh], file)
                    file_counter += 1
                    
                if obj_counter >= sample_ratio[0] and bg_counter == sample_ratio[1]:  # control the ratio between 2 types
                    obj_counter -= sample_ratio[0]
                    bg_counter = 0


class RCNN_classifier_Dataset(torch.utils.data.Dataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, dataset, cfg,
                 data_batch_size=200,
                 IoU_threshold={'positive':0.5, 'partial':0.3},
                 sample_ratio=(32, 96),
                 data_path='../'):
        """
        Args:
            label_file (list of tuple(im path, label)): Path to image + annotations.
            im_root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.dataset = dataset
        self.transform = transforms.Compose([ # preprocess image
            transforms.Resize((cfg['image_size'], cfg['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.dataset_exists()==False:
            self.generate_dataset(sample_ratio, IoU_threshold)
        else: 
            print('[*] Loading dataset from', self.data_path)
            with open(self.data_path + 'train_images_classifier.pkl', 'rb') as f:
                self.train_images = pickle.load(f)
            with open(self.data_path + 'train_labels_classifier.pkl', 'rb') as f:
                self.train_labels = pickle.load(f)

            # check if both files are complete, flawless
            if not len(self.train_images)==len(self.train_labels):
                raise ValueError('The loaded dataset is invalid (of different size).')

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.fromarray(cv2.cvtColor(self.train_images[idx], cv2.COLOR_BGR2RGB))
        return {'image': self.transform(image),
                'label': self.train_labels[idx][0],
                'est_bbox': self.train_labels[idx][1],
                'gt_bbox': self.train_labels[idx][2]}

    '''
    # not working when interact too much w/ drive :(
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.im_root_dir, self.label_file[idx][0])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return  {'image': image, 'label': self.label_file[idx][1]}
    '''
    def dataset_exists(self):
        if os.path.exists(self.data_path+'train_images_classifier.pkl')==False:
            return False
        if os.path.exists(self.data_path + 'train_labels_classifier.pkl')==False:
            return False
        
        return True    

    def generate_dataset(self, sample_ratio, IoU_threshold, padding=16):
        #https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

        image_dir=self.dataset.train_dir+'/JPEGImages/'
        annot_dir=self.dataset.train_dir+'/Annotations/'
        obj_counter = 0
        self.train_images=[]
        self.train_labels=[]

        print('[*] Generating classifier dataset for R-CNN.')

        img_list = os.listdir(image_dir)
        img_list = img_list[:5000]
        pbar = tqdm(img_list, position=0, leave=True)  # only 200 images
        
        for img_name in pbar:   
            pbar.set_description(f"Data size: {len(self.train_labels)}")

            # load image & gt bounding boxes 
            image = cv2.imread(image_dir + img_name)
            xml_path=annot_dir+img_name[:-4]+'.xml'
            try:
                gt_bboxes = self.dataset.read_xml(xml_path)
            except:
                continue

            # directly use gt bboxes as positive samples
            for gt_bbox in gt_bboxes:
                cropped = image[gt_bbox['y1']:gt_bbox['y2'], gt_bbox['x1']:gt_bbox['x2'], :]
                self.train_images.append(cropped)

                gt_bbox_xywh = convert_from_corners_to_midpoint(gt_bbox)
                est_bbox_xywh = gt_bbox_xywh
                self.train_labels.append([gt_bbox['class'], est_bbox_xywh, gt_bbox_xywh])
            obj_counter += len(gt_bboxes)

            # time to collect background :)
            if obj_counter >= sample_ratio[0]:
                obj_counter -= sample_ratio[0]
                bg_counter = 0

                # generete bbox proposals via selective search
                rects = selective_search(image)[:2000]  # parse first 2000 boxes
                random.shuffle(rects)

                # loop through all ss bounding box proposals
                for (x, y, w, h) in rects:
                    # apply padding
                    x1, x2 = np.clip([x-padding, x+w+padding], 0, image.shape[1])
                    y1, y2 = np.clip([y-padding, y+h+padding], 0, image.shape[0])
                    bbox_est = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                    is_object = False
                    
                    # check the proposal with every elements of the gt boxes
                    for gt_bbox in gt_bboxes:
                        iou = calculate_iou(gt_bbox, bbox_est)
                        if iou > IoU_threshold['partial']: # if object
                            is_object = True
                            break

                    # save image
                    if is_object == False:
                        bg_counter += 1
                        cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]
                        self.train_images.append(cropped)

                        est_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                        gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                        self.train_labels.append([0, est_bbox_xywh, gt_bbox_xywh])

                    # control the ratio between 2 types
                    if bg_counter == sample_ratio[1]:
                        break

        with open(self.data_path + 'train_images_classifier.pkl', 'wb') as f:
            pickle.dump(self.train_images, f)
        with open(self.data_path + 'train_labels_classifier.pkl', 'wb') as f:
            pickle.dump(self.train_labels, f)
        print('[*] Dataset generated! Saving labels to', self.data_path)


def RCNN_DatasetLoader(voc_dataset, cfg, training_cfg, shuffle=True):
    ds = RCNNDataset(voc_dataset, cfg)
    return DataLoader(ds, batch_size=training_cfg['batch_size'], shuffle=shuffle, num_workers=2)

def RCNN_classifier_DatasetLoader(voc_dataset, cfg, training_cfg, shuffle=True):
    ds = RCNN_classifier_Dataset(voc_dataset, cfg)
    return DataLoader(ds, batch_size=training_cfg['batch_size'], shuffle=shuffle, num_workers=2)


if __name__ == '__main__':
    config = {'image_size':224, 'n_classes':21, 'bbox_reg': True, 'network': 'efficientnet-b0', 'max_proposals':2000, 'pad': 16}
    train_config = {'log_wandb':True, 'logging': ['plot'],
                    'epochs': 5, 'batch_size':128, 'lr': 0.001, 'lr_decay':0.5, 'l2_reg': 1e-5, 'bbox_iou_threshold':0.6}
    voc_2012_classes = ['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]

    voc_dataset = VOC2012()
    loader = RCNN_DatasetLoader(voc_dataset, config, train_config)

    for x in loader:
        img = x['image']
        label = x['label']
        for idx in range(128):
            print(voc_2012_classes[label[idx]])
            np_img = img[idx].numpy().astype(np.uint8)
            plt.imshow(np.transpose(np_img, (1, 2, 0)))
            plt.show()