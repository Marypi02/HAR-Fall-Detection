import torch
import lightning as lit
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from net.cnn import ConvBlock, FeedForwardBlock, ConvAutoencoder

class Net(lit.LightningModule):
    def __init__(self,cfg):  
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.depth = cfg.depth 
        self.cfg = cfg

        # self.embed = instantiate(cfg.embed) # convoluzione deconvoluzione , loss in due parti regressione e classificazione 
        self.conv_ae = instantiate(cfg.embed) # rinomino la variabile per distinguere il caso
        self.features = nn.Sequential(*[instantiate(cfg.block) for _ in range(self.depth-1)]) # nel caso di ConvAE, è 0, disattivando gli strati intermedi e garantendo il flusso ConvAE --> LSTM
        self.lstm_block = instantiate(cfg.rnn_block)
        self.unembed = instantiate(cfg.unembed)   

        self.train_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)

        self.test_precision = MulticlassPrecision(num_classes=cfg.num_classes, average=None)
        self.test_recall = MulticlassRecall(num_classes=cfg.num_classes, average=None)
        self.test_f1score = MulticlassF1Score(num_classes=cfg.num_classes, average=None)

        self.conf_mat = MulticlassConfusionMatrix(num_classes=cfg.num_classes)

        self.class_names = [
            "Walking", "Upstairs", "Downstairs", 
            "Sitting", "Standing", "Laying"
        ]

    # --- forward for ConvAE + LSTM
    def forward(self, x):
        # 1- ENCODER --> output del bottleneck
        x = self.conv_ae(x, is_har = True)

        # 2- FLATTEN
        # Da (batch, channels, seq_len_ridotta) a (batch, seq_len_ridotta, features/channels), quindi per esempio da [batch, 128, 32] --> [batch, 32, 128]
        x = x.permute(0, 2, 1) # cambio dimensione per LSTM [batch, seq_len, features]
        
        # 3- LSTM
        x = self.lstm_block(x)  
        
        # 4- CLASSIFICAZIONE
        x = x.view(x.size(0), -1)
        x = self.unembed(x) 
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, label_smoothing=0.1) # label_smoothing dice "Non essere sicuro al 100%"
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(y_hat, y), prog_bar=True)

        self.test_precision.update(y_hat, y)
        self.test_recall.update(y_hat,y)
        self.test_f1score.update(y_hat,y)
        self.conf_mat.update(y_hat,y)

    def on_test_epoch_end(self):
        # vettore con 6 valori
        precision_score = self.test_precision.compute()
        recall_score = self.test_recall.compute()
        f1_score = self.test_f1score.compute()

        for i,name in enumerate(self.class_names):
            self.log(f"test_precison_{name}: ", precision_score[i])
            self.log(f"test_recall_{name}: ", recall_score[i])
            self.log(f"test_f1_{name}: ", f1_score[i])

        conf_mat = self.conf_mat.compute().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if isinstance(self.logger, lit.pytorch.loggers.WandbLogger):
            self.logger.experiment.log({
                "confusion_matrix": wandb.Image(plt)
            })

        plt.close()

        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1score.reset()
        self.conf_mat.reset()
        

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, self.parameters())

        # Lr scheduler per ridurre lr se la val_loss si blocca
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.5, 
            patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
