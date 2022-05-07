from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Literal, Any
import albumentations as A
from torchvision import transforms
import torch
import os 
import cv2 
import numpy as np
import torch.utils.data as data
import pytorch_lightning as pl
import pandas as pd
import warnings


from transforms import AlbuTransformCollection


class BirdsDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str, 
        dataframe: pd.DataFrame,
        transform: A.core.composition.Compose,
        _supervise_: Optional[bool] = False
    ):
        super(BirdsDataset, self).__init__()
        if _supervise_:
            warnings.warn('Warning! Using supervise mode in dataset, disable it in order to train model')
        self._supervise_ = _supervise_
        self.root = root
        self.transform = transform
        self.numpy_to_tensor_transfom = transforms.ToTensor()

        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def unique_classes(self):
        return len(set(list(self.dataframe['class index'])))

    def __getitem__(self, idx):
        element = self.dataframe.loc[idx]

        image = cv2.imread(os.path.join(self.root, element['filepaths']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = element['class index']

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        if self._supervise_:
            return image 
        
        image = self.numpy_to_tensor_transfom(image.astype(np.uint8))
 
        return image, label



class BirdsDataManger(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        img_size: Optional[int] = 224,
        grayscale: Optional[bool] = False,
        transforms_collection: Any = AlbuTransformCollection,
        _supervise_whole_dataset: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.grayscale = grayscale
        self.transforms_collection = transforms_collection
        self.batch_size = batch_size
        self._supervise_whole_dataset = _supervise_whole_dataset

        if self._supervise_whole_dataset:
            warnings.warn("Warning!\n_supervise_whole_dataset set to True,\
                \nIt is inappropriate mode to train model!")

    def unique_classes(self):
        return self.train_data.unique_classes()

    def prepare_data(self):
        print('PREPARE DATA CALL')
        self.birds_df = pd.read_csv('{}/birds.csv'.format(self.root))

    def setup(self, stage: Optional[str] = None):
        print('SETUP CALL')
        if self._supervise_whole_dataset:
            supervise_transform = self.transforms_collection.validation_transform(
                size_=self.img_size,
                to_gray=self.grayscale,
                _normalize_=False,
            )
            train_df = self.birds_df[self.birds_df['data set'] == 'train']
            self.supervise_data = BirdsDataset(
                root=self.root, 
                dataframe=train_df, 
                transform=supervise_transform,
            )
            return None
        
        train_df = self.birds_df[self.birds_df['data set'] == 'train']
        test_df = self.birds_df[self.birds_df['data set'] == 'test']
        val_df = self.birds_df[self.birds_df['data set'] == 'valid']

        train_transforms = self.transforms_collection.train_transform(
            size_=self.img_size,
            to_gray=self.grayscale,
            _normalize_=True,
        )

        val_transforms = self.transforms_collection.validation_transform(
            size_=self.img_size,
            to_gray=self.grayscale,
            _normalize_=True,
        )

        self.train_data = BirdsDataset(
            root=self.root, 
            dataframe=train_df, 
            transform=train_transforms,
        )
        self.val_data = BirdsDataset(
            root=self.root, 
            dataframe=val_df, 
            transform=val_transforms,
        )
        self.test_data = BirdsDataset(
            root=self.root, 
            dataframe=test_df, 
            transform=val_transforms,
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=os.cpu_count()
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=os.cpu_count()
        )

    def supervise_dataloder(self):
        warnings.warn('Warning! Using dataloader for supervising, not for training!')
        return data.DataLoader(
            self.supervise_data , 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=os.cpu_count()
        )
