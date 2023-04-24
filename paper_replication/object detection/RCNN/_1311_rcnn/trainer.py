import gc
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import accuracy_score

class RCNN_Trainer():
    def __init__(self, model, loader, classifier_dataloader, validator=None, device='cpu'):
        self.model = model
        self.loader = loader
        self.classifier_dataloader = classifier_dataloader
        self.validator = validator
        self.device = device

    def fine_tuning(self, train_cfg):
        if train_cfg['log_wandb']:
            wandb.init(project='rcnn', entity='krenerd77')
            wandb.watch(self.model, log_freq=100)

        loss_fn = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        optimizer = optim.Adam(self.model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['l2_reg'])
        if self.model.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.model.optimizer_state_dict)

        # lr schedule
        if 'lr_decay' in train_cfg:
            lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg['lr_decay'])
        else: # constant lr
            lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

        for epoch in range(train_cfg['epochs']):
            self.model.train()  # set model to train mode
            print('[*] Training epoch', epoch + 1, '/', train_cfg['epochs'])

            # implement training step -------------------------
            pbar = tqdm(self.loader, position=0, leave=True)
            for step, data in enumerate(pbar):
                # inference
                features = self.model.convnet(data['image'].to(self.device))
                features = self.model.flatten(features)
                output = self.model.classifier(features)

                # backprop
                clf_loss = loss_fn(output, data['label'].to(self.device))
                loss = clf_loss

                # bbox regression loss
                if self.model.dobbox_reg:
                    bbox_est = self.model.bbox_reg(features)
                    # regression targets are described in Appendix C. 
                    p_x, p_y, p_w, p_h = data['est_bbox'][0], data['est_bbox'][1], data['est_bbox'][2], data['est_bbox'][3]
                    g_x, g_y, g_w, g_h = data['gt_bbox'][0], data['gt_bbox'][1], data['gt_bbox'][2], data['gt_bbox'][3]

                    t_x = (g_x - p_x) / p_w
                    t_y = (g_y - p_y) / p_h
                    t_w = torch.log(g_w) / p_w
                    t_h = torch.log(g_h) / p_h

                    bbox_ans = torch.stack([t_x, t_y, t_w, t_h], axis=1)
                    bbox_ans = bbox_ans.float().to(self.device)

                    # count only images that are not background
                    not_bg = (data['label']>0).reshape(len(data['label']), 1).to(self.device)  # mask about whether each image is a background
                    bbox_est = bbox_est * not_bg
                    bbox_ans = bbox_ans * not_bg

                    # add to loss    
                    bbox_loss = mse(bbox_est, bbox_ans)
                    loss += bbox_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output.cpu().detach()
                output = output.argmax(dim=1)
                print('\n', data['label'])
                print(output)

                # logging ------------------------------------------
                acc = accuracy_score(data['label'].numpy(), output.numpy())
                pbar.set_description(f"Loss: %.3f  Accuracy: %.3f" % (loss.cpu().detach().numpy(), acc))

                if train_cfg['log_wandb'] and (step + 1) % 100 == 0:
                    logdict = {}
                    logdict['clf_loss'] = clf_loss
                    logdict['accuracy'] = acc

                    if self.model.dobbox_reg:
                        logdict['bbox_loss'] = bbox_loss

                    wandb.log(logdict)

            # update lr
            lr_schedule.step()

            # # save checkpoints and log
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'RCNN_checkpoint.pt')

            if self.validator is not None: 
                self.validator.validate()

    def classifier_training(self, train_cfg):
        if train_cfg['log_wandb']:
            wandb.init(project='rcnn_classifier', entity='krenerd77')
            wandb.watch(self.model.classifier, log_freq=100)

        loss_fn = nn.CrossEntropyLoss()

        # Gradients for final classifier only
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['l2_reg'])

        for epoch in range(train_cfg['epochs']):
            self.model.train()  # set model to train mode
            print('[*] Training epoch', epoch + 1, '/', train_cfg['epochs'])

            pbar = tqdm(self.classifier_dataloader, position=0, leave=True)
            for step, data in enumerate(pbar):
                # implement training step -------------------------
                # inference
                features = self.model.convnet(data['image'].to(self.device))
                features = self.model.flatten(features)
                output = self.model.classifier(features)

                # backprop
                clf_loss = loss_fn(output, data['label'].to(self.device))
                loss = clf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output.cpu().detach()
                output = output.argmax(dim=1)
                print('\n', data['label'])
                print(output)

                # logging ------------------------------------------
                acc = accuracy_score(data['label'].numpy(), output.numpy())
                pbar.set_description(f"Loss: %.3f  Accuracy: %.3f" % (loss.cpu().detach().numpy(), acc))

                if train_cfg['log_wandb'] and (step + 1) % 100 == 0:
                    logdict = {}
                    logdict['clf_loss'] = clf_loss
                    logdict['accuracy'] = acc
                    wandb.log(logdict)

            # save checkpoints and log
            torch.save(self.model, 'RCNN_checkpoint.pt')
            if self.validator is not None:
                self.validator.validate()