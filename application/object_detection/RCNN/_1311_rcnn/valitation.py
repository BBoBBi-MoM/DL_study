import wandb
import os
import cv2
import torch
from torch.utils.data import Dataset

classes = ['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]

class PlotSamples():
    def __init__(self, to_sample=30):
        # create table for plotting 
        self.to_sample = to_sample
        self.images = []

    def log(self, model, val_dataset):
        print('[*] Plotting samples to wandb board...')
        sampled = []
        for x in range(self.to_sample):
            img = val_dataset[x]['image']
            gt_bboxes = val_dataset[x]['label']
            est_bboxes = model.inference(img)

            img = self.plot_results(img, est_bboxes)
            img = self.plot_results(img, gt_bboxes, ground_truth=True, color=(255, 0, 0))
            cv2.imshow('results', img)
            cv2.waitKey(0)

            sampled.append(wandb.Image(img))
        self.images.append(sampled)

        # write to table
        sample_columns = ['sample '+str(x+1) for x in range(self.to_sample)]
        sample_table = wandb.Table(columns=sample_columns)

        for step_image in self.images:
            sample_table.add_data(*step_image)
        wandb.run.log({"Samples_table" : sample_table})

    def plot_results(self, image, bboxes, ground_truth=False, color=(0, 69, 255)):
        bbox_thickness = 2
        bbox_color = color

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = color
        line_thickness = 1

        for box in bboxes:
            if not ground_truth:
                bbox = box['bbox']
                conf = box['conf'].item()
                x1 = bbox['x1'].int().item()
                y1 = bbox['y1'].int().item()
                x2 = bbox['x2'].int().item()
                y2 = bbox['y2'].int().item()
            else:
                bbox = box
                conf = 1
                x1 = bbox['x1']
                y1 = bbox['y1']
                x2 = bbox['x2']
                y2 = bbox['y2']

            class_name = classes[box['class']]

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
            cv2.putText(image, f"{class_name} %.2f" %(conf), (x1, y1 - 5),
                        font, font_scale, font_color, line_thickness)
        return image

class ValDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, dataset):
        self.base_dir = dataset.test_dir
        self.dataset = dataset
        self.images = sorted(os.listdir(dataset.test_dir+'/JPEGImages/'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.base_dir + '/JPEGImages/' + self.images[idx]
        image = cv2.imread(image_path)

        xml_path = self.base_dir + '/Annotations/' + self.images[idx][:-4]+'.xml'
        gt_bboxes = self.dataset.read_xml(xml_path)  # load bboxes in list of dicts {x1, x2, y1, y2, class}

        return {'image': image, 'label': gt_bboxes}
    
class Validator():
    def __init__(self, val_dataset, model, train_cfg):
        self.val_dataset = val_dataset
        self._initialize(model, train_cfg['logging'])

    def _initialize(self, model, log_keys):
        self.model = model 

        self.loggers = {}
        self.log_keys = log_keys
        log_funcs = {'plot': PlotSamples}
        for logtype in log_keys: 
            self.loggers[logtype] = log_funcs[logtype]()

    def validate(self):
        self.model.eval()
        for key in self.log_keys:
            self.loggers[key].log(self.model, self.val_dataset)