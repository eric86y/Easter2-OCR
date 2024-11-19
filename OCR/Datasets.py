import cv2
import torch
from OCR.Encoder import LabelEncoder
from torch.utils.data import Dataset
from typing import List, Optional
from albumentations.core.composition import Compose

from OCR.Utils import (
    binarize,
    pad_ocr_line
)

class CTCDataset(Dataset):
    def __init__(
        self,
        images: List[str],
        labels: List[str],
        label_encoder: LabelEncoder,
        img_height: int = 80,
        img_width: int = 2000,
        augmentations: Optional[Compose] = None,
    ):
        super(CTCDataset, self).__init__()

        self.images = images
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.label_encoder = label_encoder
        self.augmentations = augmentations
 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        if image is None:
            print(f"error reading image: {self.images[index]}")
              # grayscale
        image = binarize(image)
        
        if self.augmentations is not None:
            aug = self.augmentations(image=image)

            image = aug["image"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = pad_ocr_line(
            image, target_width=self.img_width, target_height=self.img_height
        )
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        label = self.labels[index]
        target = self.label_encoder.encode(label)
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length


def ctc_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths