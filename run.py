import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
import torch

from data import BirdsDataManger
from model import BirdsModel
from transforms import AlbuTransformCollection

# Reproducubility
import imgaug
imgaug.random.seed(42)
import random
random.seed(42)
torch.manual_seed(42)
import numpy as np
np.random.seed(0)



BATCH_SIZE = 128
IMG_SIZE = 224

CONFIGS = {
    'EfficientNetB0-from-scratch': {
        'model_name': 'EfficientNet',
        'weights': 'none',
        'family': 'b0',
        'grayscale': False
    },

    'EfficientNetB0-from-pretrained': {
        'model_name': 'EfficientNet',
        'weights': 'imagenet',
        'family': 'b0',
        'grayscale': False
    },

    'MobileNetV3-Small-from-scratch': {
        'model_name': 'MobileNetV3',
        'weights': 'none',
        'family': 'small',
        'grayscale': False
    },

    'MobileNetV3-Small-from-pretrained': {
        'model_name': 'MobileNetV3',
        'weights': 'imagenet',
        'family': 'small',
        'grayscale': False
    },

    'MobileNetV3-Large-from-scratch': {
        'model_name': 'MobileNetV3',
        'weights': 'none',
        'family': 'large',
        'grayscale': False
    },

    'MobileNetV3-Large-from-pretrained': {
        'model_name': 'MobileNetV3',
        'weights': 'imagenet',
        'family': 'large',
        'grayscale': False
    },

    'Grayscale-EfficientNetB0-from-pretrained': {
        'model_name': 'EfficientNet',
        'weights': 'imagenet',
        'family': 'b0',
        'grayscale': True
    },

    'Grayscale-EfficientNetB0-from-scratch': {
        'model_name': 'EfficientNet',
        'weights': 'none',
        'family': 'b0',
        'grayscale': True
    },

    'Grayscale-MobileNetV3-Small-from-pretrained': {
        'model_name': 'MobileNetV3',
        'weights': 'imagenet',
        'family': 'small',
        'grayscale': True
    },

    'Grayscale-MobileNetV3-Small-from-scratch': {
        'model_name': 'MobileNetV3',
        'weights': 'none',
        'family': 'small',
        'grayscale': True
    },

    'Grayscale-MobileNetV3-Large-from-pretrained': {
        'model_name': 'MobileNetV3',
        'weights': 'imagenet',
        'family': 'large',
        'grayscale': True
    },

    'Grayscale-MobileNetV3-Large-from-scratch': {
        'model_name': 'MobileNetV3',
        'weights': 'none',
        'family': 'large',
        'grayscale': True
    },
}



def run():
    for key in CONFIGS:
        print('\n\n\nStarting {}\n\n\n'.format(key))
        run_name = key
        logger = TensorBoardLogger('lightning_logs', name=run_name)

        dm = BirdsDataManger(
            root='./dataset/',
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            grayscale=CONFIGS[key]['grayscale'],
            transforms_collection=AlbuTransformCollection,
        )
        dm.prepare_data()
        dm.setup()

        model_kwargs = {
            'n_classes': dm.unique_classes(),
            'weights': CONFIGS[key]['weights'],
            'family': CONFIGS[key]['family']
        }
        model = BirdsModel(CONFIGS[key]['model_name'], **model_kwargs)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            dirpath='runs/{}/'.format(run_name),
            filename='birds-adventures-{epoch:02d}-{step:d}-{val_loss:.4f}',
            save_last=True,
            verbose=True,
            every_n_epochs=1
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=5,
            verbose=True,
            strict=True,
        )
        weight_averaging_callback = StochasticWeightAveraging(swa_lrs=1e-2)

        trainer = pl.Trainer(
            gpus=[1],
            precision=16,
            # accumulate_grad_batches=5,
            max_epochs=50,
            min_epochs=1,
            progress_bar_refresh_rate=1,
            callbacks=[checkpoint_callback, early_stopping_callback, weight_averaging_callback],
            logger=logger,
            log_every_n_steps=1,
            flush_logs_every_n_steps=50,
        )

        # Train the model 
        torch.cuda.empty_cache()
        trainer.fit(model, dm)
        result = trainer.test(model, dm)


if __name__ == '__main__':
    run()
