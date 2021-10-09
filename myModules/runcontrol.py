###
# file name: runcontrol.py (Wavemotion-lightning)
# Author: Dr. Xiaoxian Guo @ Shanghai Jiao Tong University, China
# Email: xiaoxguo@sjtu.edu.cn
# Website: https://naoce.sjtu.edu.cn/teachers/9004.html
# Github: https://github.com/XiaoxG/waveMotion-lightning/
# Create date: 2021/10/08
# Wavemotion-lightning-- a demonstration code for wave motion prediction of an offshore platform. 
# Refers to preprint Arvix: "Probabilistic prediction of the heave motions of a semi-submersible by a deep learning model"
###

import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchmetrics import ExplainedVariance
from myModules.model import NetworkRNN

class LitModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.lr = params.training.lr
        self.model = NetworkRNN(params, self.device)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.save_hyperparameters(params)
        self.ev_train = ExplainedVariance()
        self.ev_val = ExplainedVariance()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        if self.model.rnn_type == 'gru':
            self.model.hidden_cell_input.detach_()
            if self.model.hidden_cell_layers != None:
                self.model.hidden_cell_layers.detach_()
        elif self.model.rnn_type == 'lstm':
            self.model.hidden_cell_input1.detach_()
            self.model.hidden_cell_input2.detach_()
            if self.hparams.model.num_rnn_layers > 1:
                self.model.hidden_cell_layers1.detach_()
                self.model.hidden_cell_layers2.detach_()

        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.ev_train(preds, y)
        
        self.log('train_loss', loss)
        self.log('train_acc', self.ev_train, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        milestones=[10, 30, 50, 100]
        lr_scheduler = {'scheduler': optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1),
                     'interval': 'epoch'  # 'step' or 'epoch'
                    }
        return [optimizer], [lr_scheduler]
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.ev_val(preds, y)

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', self.ev_val, prog_bar=True, on_step = False, on_epoch = True, sync_dist=True)
