import cv2
import os
import random

import tarfile
import urllib.request
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class VOCDataset():
    def __init__(self):
        self.train_dir=None
        self.test_dir=None
        self.trainDataLink=None
        self.testDataLink=None

        self.common_init()

    def common_init(self):
        # init that must be shared among all subclasses of this method
        self.label_type=['none','aeroplane',"Bicycle",'bird',"Boat","Bottle","Bus","Car","Cat","Chair",'cow',"Diningtable","Dog","Horse","Motorbike",'person', "Pottedplant",'sheep',"Sofa","Train","TVmonitor"]
        self.convert_id=['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]
        self.convert_labels={}
        for idx, x in enumerate(self.label_type):
            self.convert_labels[x.lower()]=idx

        self.num_classes = len(self.label_type) # 20 + 1(none)

    def download_dataset(self, validation_size=5000):
        # download voc train dataset
        print('[*] Downloading dataset...')
        print(self.trainDataLink)
        urllib.request.urlretrieve(self.trainDataLink, 'voctrain.tar')

        print('[*] Extracting dataset...')
        tar = tarfile.open('voctrain.tar', "r:")
        tar.extractall('./object_detection/content/VOCtrain')
        tar.close()
        # os.remove('voctrain.tar')

        if self.testDataLink is None: 
            # move 5K images to validation set
            print('[*] Moving validation data...')
            ensure_dir(self.test_dir+'/Annotations/')
            ensure_dir(self.test_dir+'/JPEGImages/')

            random.seed(42)
            val_images = random.sample(sorted(os.listdir(self.train_dir + '/JPEGImages')), validation_size)

            for path in val_images:
                img_name = path.split('/')[-1].split('.')[0]
                # move image
                os.rename(self.train_dir+'/JPEGImages/'+img_name+'.jpg', self.test_dir+'/JPEGImages/'+img_name+'.jpg')
                # move annotation
                os.rename(self.train_dir+'/Annotations/'+img_name+'.xml', self.test_dir+'/Annotations/'+img_name+'.xml')
        else: 
            # Load from val data
            print('[*] Downloading validation dataset...')
            urllib.request.urlretrieve(self.testDataLink, 'voctest.tar')

            print('[*] Extracting validation dataset...')
            tar = tarfile.open('voctest.tar', "r:")
            tar.extractall('./object_detection/content/VOCtest')
            tar.close()
            #   os.remove('voctest.tar')

    def read_xml(self, xml_path): 
        object_list=[]

        tree = ET.parse(open(xml_path, 'r'))
        root = tree.getroot()

        objects = root.findall("object")
        for _object in objects:
            name = _object.find("name").text
            bndbox = _object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            class_name = _object.find('name').text
            object_list.append({'x1': xmin, 'x2': xmax, 'y1': ymin, 'y2': ymax, 'class': self.convert_labels[class_name]})

        return object_list

class VOC2007(VOCDataset):
    def __init__(self):
        self.train_dir = './content/VOCtrain/VOCdevkit/VOC2007'
        self.test_dir = './content/VOCtest/VOCdevkit/VOC2007'
        self.trainDataLink = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
        self.testDataLink = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
        self.common_init()  # mandatory
        
class VOC2012(VOCDataset):
    def __init__(self):
        self.train_dir = './content/VOCtrain/VOCdevkit/VOC2012'
        self.test_dir = './content/VOCtest/VOCdevkit/VOC2012'
        # original site goes down frequently, so we use a link to the clone alternatively
        # self.trainDataLink='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' 
        self.trainDataLink = 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
        self.testDataLink = None
        self.common_init()  # mandatory


if __name__ == '__main__':
    voc_dataset = VOC2012()
    # voc_dataset.download_dataset()

    sorted_img_list = sorted(os.listdir(voc_dataset.train_dir+'/JPEGImages'))
    for i in range(100):
        img_name = sorted_img_list[i][:-4]
        img = cv2.imread(voc_dataset.train_dir+'/JPEGImages/'+img_name+'.jpg')
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
        print('Shape:', img.shape)
        
        xml_path=voc_dataset.train_dir+'/Annotations/'+img_name+'.xml'
        tree = ET.parse(open(xml_path, 'r'))
        root=tree.getroot()

        # plot bounding boxes
        box_img = img.copy()
        bbox_color = (0, 69, 255) # (b, g, r) not (r, g, b)
        bbox_thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = bbox_color
        lineType = 1

        objects = root.findall("object")
        for _object in objects:
            name = _object.find("name").text
            bndbox = _object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            class_name = _object.find('name').text

            cv2.rectangle(box_img, (xmin, ymin), (xmax, ymax), bbox_color, bbox_thickness)
            cv2.putText(box_img, class_name, (xmin, ymin-5), font, 
                fontScale,
                fontColor,
                lineType)

            cv2.imshow('bbox_img', box_img)
            cv2.waitKey(0)