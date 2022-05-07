import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import torch.optim as optim 
import torch
import numpy as np
from typing import Optional, Literal
from torchvision import models


class EfficientNetBackbone:
    @staticmethod
    def load_model(
            n_classes: int,
            weights: Literal['none', 'imagenet'],
            family: Optional[str] = 'b0'
    ):
        model = None
        if weights == 'imagenet':
            model = EfficientNet.from_pretrained('efficientnet-{}'.format(family), num_classes=n_classes)
        elif weights == 'none':
            original_dropout_rate = 0.5 * int(family[1]) / 7.0
            model = EfficientNet.from_name(
                'efficientnet-{}'.format(family),
                num_classes=n_classes,
                dropout_rate=original_dropout_rate,
                # drop_connect_rate=0.2,
                batch_norm_momentum=0.90
            )

        return model


class MobileNetV3:
    @staticmethod
    def load_model(
        n_classes: int,
        weights: Literal['none', 'imagenet'],
        family: Literal['small', 'large']
    ):
        model = None
        if family == 'small':
            if weights == 'imagenet':
                model = models.mobilenet_v3_small(pretrained=True)
            elif weights == 'none':
                model = models.mobilenet_v3_small(pretrained=False)
            
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=576, out_features=400, bias=True),
                torch.nn.Hardswish(),
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=400, out_features=n_classes, bias=True)
            )
        elif family == 'large':
            if weights == 'imagenet':
                model = models.mobilenet_v3_large(pretrained=True)
            elif weights == 'none':
                model = models.mobilenet_v3_large(pretrained=False)
            
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=960, out_features=400, bias=True),
                torch.nn.Hardswish(),
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=400, out_features=n_classes, bias=True)
            )

        return model


def adapter(model_name, **model_kwargs) -> torch.nn.Module:
    if model_name == 'EfficientNet':
        return EfficientNetBackbone.load_model(**model_kwargs)
    if model_name == 'MobileNetV3':
        return MobileNetV3.load_model(**model_kwargs)


class BirdsModel(pl.LightningModule):
    def __init__(
        self, 
        model_name: Literal[
            'EfficientNet',
            'MobileNetV3'
        ],
        **model_kwargs
    ):
        super(BirdsModel, self).__init__()
        self.backbone = adapter(model_name, **model_kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        x = self.backbone(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.3, 
            patience=0, 
            threshold=1e-2, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0, 
            eps=1e-08, 
            verbose=True
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return [optimizer], [lr_dict]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim=1) == y) / y.shape[0]

        logs = {'train_loss': loss.detach().cpu().numpy(), 'train_acccuracy': accuracy.detach().cpu().numpy()}

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acccuracy', accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss, 'acccuracy': accuracy.detach(), 'log': logs}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim=1) == y) / y.shape[0]

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        return {'val_loss': loss, 'val_accuracy': accuracy.detach()}


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim=1) == y) / y.shape[0]

        self.log("test_loss", loss, on_step=True, on_epoch=False)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=False)

        return {'test_loss': loss, 'test_accuracy': accuracy}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_acc}

        print('\n\nVAL Accuracy: {}\nVAL Loss: {}\n'.format(
            round(float(avg_acc), 3), 
            avg_loss
        ))

        self.log('val_accuracy', avg_acc, on_epoch=True, on_step=False)
        self.log('val_loss', avg_loss, on_epoch=True, on_step=False)

        return {'val_loss': avg_loss, 'val_accuracy': avg_acc, 'log': tensorboard_logs}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'test_accuracy': avg_acc}

        self.log('test_accuracy', avg_acc, on_epoch=True, on_step=False)

        return {'test_loss': avg_loss, 'test_accuracy': avg_acc, 'log': tensorboard_logs}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acccuracy'] for x in outputs]).mean()

        self.log('train_acccuracy', avg_acc, on_epoch=True, on_step=False, logger=True)
        self.log('train_loss', avg_loss, on_epoch=True, on_step=False, logger=True)

        return None