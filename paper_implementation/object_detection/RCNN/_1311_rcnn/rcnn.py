import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import TensorDataset, DataLoader

from utils import *
from rcnn_dataset import selective_search

from torchvision.models import resnet50


def RCNN(cfg, load_path=None, device='cpu'):
    if load_path:
        try:
            print('[*] Attempting to load model from:', load_path)
            return _RCNN(cfg, device, load_path)
        except:
            import traceback
            print(traceback.format_exc())
            print('[*] Model does not exist or is corrupted. Creating new model...')
            return _RCNN(cfg, device)

    else:
        print('[*] Creating model...')
        return _RCNN(cfg, device)
        
class _RCNN(nn.Module):
    def __init__(self, cfg, device='cpu', load_path=None):
        super(_RCNN, self).__init__()

        self.device = device
        self.num_classes = cfg['n_classes']
        self.dobbox_reg = cfg['bbox_reg']
        self.max_proposals = cfg['max_proposals']  # maximum number of regions to extract from given image at inference
        self.image_size = cfg['image_size']  # efficientnet-b0: 224

        self.transforms = transforms.Compose([  # transform image
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._initialize_weights()

        self.optimizer_state_dict = None
        if load_path is not None:
            check_point = torch.load(load_path)
            self.load_state_dict(check_point['model_state_dict'])

            if 'optimizer_state_dict' in check_point:
                self.optimizer_state_dict = check_point['optimizer_state_dict']

    def forward(self, x):
        features = self.convnet(x)
        features = self.flatten(features)
        pred = self.classifier(features)
        return pred, features

    def region_proposal(self, image, rgb=False):
        # convert rgb to bgr for selective search
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image

        # perform selective search to find region proposals
        rects = selective_search(image)

        proposals = []
        boxes = []
        for (x, y, w, h) in rects[:self.max_proposals]:
            roi = cv2.cvtColor(image[y:y + h, x:x + w, :], cv2.COLOR_BGR2RGB)
            roi = self.transforms(roi)

            proposals.append(roi)
            boxes.append({'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})

        return proposals, boxes

    def inference_single(self, image, rgb=False, batch_size=16, apply_nms=True, nms_threshold=0.2, silent_mode=False):
        # image shape (H, W, C)
        # image must be loaded in BGR format(cv2.imread) or else rgb must be set to True
        # https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/

        self.eval()

        # region proposal
        proposals, boxes = self.region_proposal(image, rgb)

        # convert to DataLoader for batching
        proposals = torch.stack(proposals)
        proposals = TensorDataset(proposals)
        proposals = DataLoader(proposals, batch_size=batch_size)

        # predict probability of each box
        cnt = 0
        pred_bboxes = []
        for proposal_batch in tqdm(proposals, position=0, disable=silent_mode):
            proposal_batch = proposal_batch[0].to(self.device)

            with torch.no_grad():
                pred, features = self(proposal_batch)
                pred = torch.nn.functional.softmax(pred)

            if self.dobbox_reg: 
                bbox_reg_pred = self.bbox_reg(features)

            not_bg_preds = torch.where(pred.argmax(dim=1) > 0)  # patches which are not classified bg(0)
            for idx in not_bg_preds[0]:  # loop through each image
                idx = idx.item()
                estimate = {}

                class_prob = pred[idx].cpu().numpy()
                estimate['class'] = class_prob.argmax(0)
                estimate['conf'] = class_prob.max(0)

                original_bbox = boxes[cnt * batch_size + idx]
                if self.dobbox_reg == False:
                    estimate['bbox'] = original_bbox
                else: 
                    estimate['bbox'] = self.refine_bbox(original_bbox, bbox_reg_pred[idx])

                #print(estimate)
                pred_bboxes.append(estimate)
            cnt += 1

        # apply non-max suppression to remove duplicate boxes
        if apply_nms: 
            pred_bboxes = nms(pred_bboxes, nms_threshold)

        return pred_bboxes

    def inference(self, images, rgb=False, batch_size=16, apply_nms=True, nms_threshold=0.2):
        # when given single image
        if type(images) == np.ndarray and len(images.shape) == 3:
            return self.inference_single(images, rgb, batch_size, apply_nms)

        bboxes = []
        for image in tqdm(images, position=0):
            pred_bboxes = self.inference_single(image, rgb, batch_size, apply_nms, silent_mode=True)
            bboxes.append(pred_bboxes)
        return bboxes

    def refine_bbox(self, bbox, pred): 
        # refine bbox in list of [ {'x1', 'x2', 'y1', 'y2'}, ... ]
        # pred is array of predicted refinements of shape (batch size, 4)
        x, y = (bbox['x1'] + bbox['x2']) / 2, (bbox['y1'] + bbox['y2']) / 2
        w, h = bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']

        newx = x + w * pred[0]
        newy = y + h * pred[1]
        neww = w * torch.exp(pred[2])
        newh = h * torch.exp(pred[3])

        return {'x1': newx - neww/2, 'x2': newx + neww / 2, 'y1': newy - newh/2, 'y2': newy + newh / 2}

    def _initialize_weights(self):
        print('[*] Initializing new network...')

        self.resnet = resnet50(pretrained=True)
        self.convnet = nn.Sequential(
                self.resnet.conv1,
                self.resnet.relu,
                self.resnet.maxpool,
                self.resnet.layer1,
                self.resnet.layer2,
                self.resnet.layer3,
                self.resnet.layer4
        )

        for param in self.convnet.parameters():
            param.requires_grad = False

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features//2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.resnet.fc.in_features//2, 21)
        )

        if self.dobbox_reg:
            self.bbox_reg = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features//2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.resnet.fc.in_features//2, 4)
        )

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'image_size': 224, 'n_classes': 21, 'bbox_reg': True, 'network': 'efficientnet-b0', 'max_proposals': 2000, 'pad': 16}

    x = np.random.randn(480, 640, 3).astype(np.uint8)
    model = RCNN(config, load_path='RCNN_checkpoint.pt', device=device).to(device)
    y = model.inference(x)
    print(y)