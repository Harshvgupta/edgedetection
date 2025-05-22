from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Resize, Normalize
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return Compose([
        Resize(640, 640),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_val_transforms():
    return Compose([
        Resize(640, 640),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
