import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
from pytorchcv.model_provider import get_model
import torchmetrics.functional as metrics
import numpy as np
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision
from PIL import Image
import skimage
from skimage.segmentation import slic
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from timm.models import create_model
import math


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
               
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )       

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )     

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.res(x)
        return x1+x2

class MultiModel(pl.LightningModule):
    def __init__(self, architecture, model1, model2, loss_func=nn.L1Loss()):
        super().__init__()

        self.model1=model1
        self.model2=model2
        self.loss_func = loss_func
        self.out1 = nn.Sequential(nn.Linear(384,128),nn.Linear(128,2))
        self.out2 = nn.Sequential(nn.Linear(384,128),nn.Linear(128,2))
	self.conv=ResBlock(1,32)
        self.gap = nn.AvgPool2d(2,2)

        
        self.lr = 1e-3
        self.lr_patience = 5
        self.lr_min = 1e-7

        self.glabels_p = []
        self.glabels_gt = []
        
        self.tr_loss = []
        self.vl_loss = []
        self.ts_loss = []
        self.tr_maeg = []
        self.vl_maeg = []
        self.ts_maeg = []
        
        self.olabels_p = []
        self.olabels_gt = []
        
        
        self.tr_maeo = []
        self.vl_maeo = []
        self.ts_maeo = []
        
        
    def forward(self, images):
        batch_size = images.size()[0]
        cls1=self.model1.model.forward_features(images)
        cls2=self.model2.model.forward_features(images)
        cls=torch.cat((cls1,cls2),dim =1)
        out=self.conv((cls.unsqueeze(0)).unsqueeze(0))
        out = self.gap(out)
        y1= self.out1(out)
        y2= self.out2(out)
        
        return y1,y2

    def training_step(self, batch, batch_idx):
        images, glabels, olabels = batch
        output1, output2 = self.forward(images)
        output1=output1[:,0]+output1[:,1]
        output2=output2[:,0]+output2[:,1]
        loss = ((self.loss_func(output1, glabels)))+(0.3*(self.loss_func(output2,olabels)))
        maeg = metrics.mean_absolute_error(output1, glabels)
        maeo = metrics.mean_absolute_error(output2, olabels)
        pcg = metrics.pearson_corrcoef(output1, glabels)
        pco = metrics.pearson_corrcoef(output2, olabels)
        return {'loss': loss, 'maeg': maeg, 'maeo': maeo, "pcg":pcg, "pco": pco}

    def validation_step(self, batch, batch_idx):
        images, glabels, olabels = batch
        glabels = glabels#.view(-1,1)
        olabels = olabels#.view(-1,1)
        output1, output2 = self.forward(images)
        output1=output1[:,0]+output1[:,1]
        output2=output2[:,0]+output2[:,1]
        loss = ((self.loss_func(output1, glabels)))+(0.3*(self.loss_func(output2,olabels)))
        maeg = metrics.mean_absolute_error(output1, glabels)
        maeo = metrics.mean_absolute_error(output2, olabels)
        pcg = metrics.pearson_corrcoef(output1, glabels)
        pco = metrics.pearson_corrcoef(output2, olabels)
        return {'loss': loss, 'maeg': maeg, 'maeo': maeo, "pcg":pcg, "pco": pco}

    def test_step(self, batch, batch_idx):
        images, glabels, olabels = batch
        glabels = glabels#.view(-1,1)
        olabels = olabels#.view(-1,1)
        output1, output2 = self.forward(images)
        output1=output1[:,0]+output1[:,1]
        output2=output2[:,0]+output2[:,1]
        loss = ((self.loss_func(output1, glabels)))+(0.3*(self.loss_func(output2,olabels)))
        maeg = metrics.mean_absolute_error(output1, glabels)
        maeo = metrics.mean_absolute_error(output2, olabels)
        pcg = metrics.pearson_corrcoef(output1, glabels)
        pco = metrics.pearson_corrcoef(output2, olabels)
        self.glabels_p = self.glabels_p + output1.squeeze().tolist()
        self.olabels_p = self.olabels_p + output2.squeeze().tolist()
        self.glabels_gt = self.glabels_gt + glabels.squeeze().tolist()
        self.olabels_gt = self.olabels_gt + olabels.squeeze().tolist()
                     
        return {'loss': loss, 'maeg': maeg, 'maeo': maeo, "pcg":pcg, "pco": pco}
        
      

    def training_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        maeg = torch.stack([x['maeg'] for x in outs]).mean()
        maeo = torch.stack([x['maeo'] for x in outs]).mean()
        pcg = torch.stack([x['pcg'] for x in outs]).mean()
        pco = torch.stack([x['pco'] for x in outs]).mean()
        self.tr_loss.append(loss)
        self.tr_maeg.append(maeg)
        self.tr_maeo.append(maeo)
        self.log('Loss/Train', loss, prog_bar=True, on_epoch = True)
        self.log('MAEg/Train', maeo, prog_bar=True, on_epoch = True)
        self.log('MAEo/Train', maeo, prog_bar=True, on_epoch = True)
        self.log('PCg/Train', pcg, prog_bar=True, on_epoch = True)
        self.log('PCo/Train', pco, prog_bar=True, on_epoch = True)

    def validation_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        maeg = torch.stack([x['maeg'] for x in outs]).mean()
        maeo= torch.stack([x['maeo'] for x in outs]).mean()
        pcg = torch.stack([x['pcg'] for x in outs]).mean()
        pco = torch.stack([x['pco'] for x in outs]).mean()
        self.vl_loss.append(loss)
        self.vl_maeg.append(maeo)
        self.vl_maeo.append(maeo)
        self.log('Loss/Val', loss, prog_bar=True, on_epoch = True)
        self.log('MAEg/Val', maeg, prog_bar=True, on_epoch = True)
        self.log('MAEo/Val', maeo, prog_bar=True, on_epoch = True)
        self.log('PCg/Val', pcg, prog_bar=True, on_epoch = True)
        self.log('PCo/Val', pco, prog_bar=True, on_epoch = True)
        

    def test_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        maeg = torch.stack([x['maeg'] for x in outs]).mean()
        maeo = torch.stack([x['maeo'] for x in outs]).mean()
        pcg = torch.stack([x['pcg'] for x in outs]).mean()
        pco = torch.stack([x['pco'] for x in outs]).mean()
        mae_sdvg = torch.stack([x['maeg'] for x in outs]).std()
        mae_sdvo = torch.stack([x['maeo'] for x in outs]).std()
        self.ts_loss.append(loss)
        self.ts_maeg.append(maeg)
        self.ts_maeo.append(maeo)
        self.log('Loss/Test', loss, prog_bar=True, on_epoch = True)
        self.log('PCg/Test', pcg, prog_bar=True, on_epoch = True)
        self.log('MAEg/Test', maeg, prog_bar=True, on_epoch = True)
        self.log('PCo/Test', pco, prog_bar=True, on_epoch = True)
        self.log('MAEo/Test', maeo, prog_bar=True, on_epoch = True)
        self.log('MAE_SDVg/Test',mae_sdvg, prog_bar=True, on_epoch = True)
        self.log('MAE_SDVo/Test',mae_sdvo, prog_bar=True, on_epoch = True)
      


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience, min_lr=self.lr_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'Loss/Val', "interval": 'epoch'}