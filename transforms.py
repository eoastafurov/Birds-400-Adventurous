from signal import valid_signals
import albumentations as A
import cv2
from typing import Optional


NORMALIZATION_PARAMS = {
    'color mean': (0.4704, 0.4670, 0.3900),
    'color std': (0.2390, 0.2328, 0.2543),
    'gray mean': (0.4593, 0.4593, 0.4593),
    'gray std': (0.2269, 0.2269, 0.2269)
}

class TransformCollection:
    @staticmethod
    def train_transform():
        pass

    @staticmethod
    def validation_transform():
        pass


class AlbuTransformCollection(TransformCollection):
    @staticmethod
    def train_transform(
        size_: int, 
        to_gray: bool,
        _normalize_: Optional[bool] = True
    ):  
        _togray_transform = A.augmentations.transforms.ToGray(p=1.0)
    
        _normalization = A.Normalize(
            mean=NORMALIZATION_PARAMS['color mean' if not to_gray else 'gray mean'], 
            std=NORMALIZATION_PARAMS['color std' if not to_gray else 'gray std']
        )

        train_transforms_list = [
            A.Rotate(
                limit=60, 
                p=0.5
            ),
            A.HorizontalFlip(
                p=0.5
            ),
            A.ISONoise(
                color_shift=(0.01, 0.3), 
                intensity=(0.01, 0.2), 
                always_apply=False, 
                p=0.25
            ),
            A.OpticalDistortion(
                distort_limit=0.5, 
                shift_limit=0.5, 
                interpolation=1, 
                border_mode=4, 
                value=None, 
                mask_value=None, 
                always_apply=False, 
                p=0.25
            ),
            A.GaussNoise(
                var_limit=(10.0, 200.0), 
                mean=0, 
                per_channel=True, 
                always_apply=False, 
                p=0.25
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.3, 
                brightness_by_max=False, 
                always_apply=False, 
                p=0.5
            ),
            A.augmentations.transforms.CLAHE(
                clip_limit=3.0, 
                tile_grid_size=(8, 8),
                always_apply=False, 
                p=0.1
            ),
            A.augmentations.transforms.Downscale(
                scale_min=0.3, 
                scale_max=0.75, 
                interpolation=0, 
                always_apply=False, 
                p=0.25
            ),
            A.augmentations.transforms.FancyPCA(
                alpha=0.2, 
                always_apply=False, 
                p=0.25
            ),
            A.augmentations.transforms.Sharpen(
                alpha=(0.2, 0.5), 
                lightness=(0.5, 1.0), 
                always_apply=False, 
                p=0.25
            ),
            A.augmentations.CoarseDropout(
                max_holes=8, 
                max_height=size_//3, 
                max_width=size_//3, 
                min_holes=1, 
                min_height=size_//10, 
                min_width=size_//10, 
                fill_value=0, 
                mask_fill_value=None, 
                always_apply=False, 
                p=0.25
            ),
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            ),
        ]

        if to_gray:
            train_transforms_list.append(_togray_transform)

        if _normalize_:
            train_transforms_list.append(_normalization)

        train_transforms = A.Compose(train_transforms_list)

        return train_transforms

    @staticmethod
    def validation_transform(
        size_: int, 
        to_gray: bool,
        _normalize_: Optional[bool] = True
    ):
        _togray_transform = A.augmentations.transforms.ToGray(p=1.0)
    
        _normalization = A.Normalize(
            mean=NORMALIZATION_PARAMS['color mean' if not to_gray else 'gray mean'], 
            std=NORMALIZATION_PARAMS['color std' if not to_gray else 'gray std']
        )
        
        val_transforms_list = [
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            )
        ]

        if to_gray:
            val_transforms_list.append(_togray_transform)

        if _normalize_:
            val_transforms_list.append(_normalization)

        val_transforms = A.Compose(val_transforms_list)

        return val_transforms
