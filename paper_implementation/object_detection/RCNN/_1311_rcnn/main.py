'https://arxiv.org/pdf/1311.2524.pdf'
'https://medium.com/codex/implementing-r-cnn-object-detection-on-voc2012-with-pytorch-b05d3c623afe'
'https://colab.research.google.com/drive/1nCj54XryHcoMARS4cSxivn3Ci1I6OtvO?usp=sharing#scrollTo=B6LEi2IWWgLu'
'https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb'
'https://herbwood.tistory.com/6'

import torch
from pascal_voc import VOC2012
from rcnn_dataset import RCNN_DatasetLoader, RCNN_classifier_DatasetLoader
from rcnn import RCNN
from valitation import Validator, ValDataset
from trainer import RCNN_Trainer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'image_size': 224, 'n_classes': 21, 'bbox_reg': True, 'network': 'efficientnet-b0', 'max_proposals': 2000, 'pad': 16}
    train_config = {'log_wandb': False, 'logging': ['plot'],
                    'epochs': 5, 'batch_size': 128, 'lr': 0.001, 'lr_decay': 0.5, 'l2_reg': 1e-5, 'bbox_iou_threshold': 0.6}

    voc_dataset = VOC2012()
    val_dataset = ValDataset(voc_dataset)
    
    model = RCNN(config, load_path='RCNN_checkpoint.pt', device=device).to(device)
    validator = Validator(val_dataset, model, train_config)

    loader = RCNN_DatasetLoader(voc_dataset, config, train_config)
    # trainer = RCNN_Trainer(model, loader, None, validator, device=device)
    trainer = RCNN_Trainer(model, loader, None, None, device=device)
    trainer.fine_tuning(train_config)
    loader = None

    # classifier_dataloader = RCNN_classifier_DatasetLoader(voc_dataset, config, train_config)
    # # trainer = RCNN_Trainer(model, None, classifier_dataloader, validator, device=device)
    # trainer = RCNN_Trainer(model, None, classifier_dataloader, None, device=device)
    # trainer.classifier_training(train_config)
    # classifier_dataloader = None

    validator.validate()