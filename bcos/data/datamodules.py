import os
import time
from typing import List, Optional, Tuple

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_info
except ImportError:
    raise ImportError(
        "Please install pytorch-lightning for using data modules: "
        "`pip install pytorch-lightning`"
    )

import pickle
import random

import pandas as pd
import numpy as np
import pydicom
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from torchvision.datasets import CIFAR10, ImageFolder

import bcos.settings as settings

from .categories import CIFAR10_CATEGORIES, IMAGENET_CATEGORIES
from .cc3m import CC3MImg, CC3MText, CustomDataCollatorImg, CustomDataCollatorText
from .sampler import RASampler
from .transforms import RandomCutmix, RandomMixup, SplitAndGrid

__all__ = [
    "ImageNetDataModule",
    "CIFAR10DataModule",
    "ClassificationDataModule",
    "VOCDataModule",
    "CC3MDataModule",
    "PneumoniaDataModule",
    "VinBigXrayDataModule",
]

def get_random_cut(dataset, cut_ratio):
    all_indices = [*range(0, len(dataset))]
    cut_idx = int(cut_ratio* len(all_indices))
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_indices) #Always shuffle the same way.
    random.setstate(state)
    return all_indices[:cut_idx], all_indices[cut_idx:]

class ClassificationDataModule(pl.LightningDataModule):
    """Base class for data modules for classification tasks."""

    NUM_CLASSES: int = None
    """Number of classes in the dataset."""
    NUM_TRAIN_EXAMPLES: int = None
    """Number of training examples in the dataset. Need not be defined."""
    NUM_EVAL_EXAMPLES: int = None
    """Number of evaluation examples in the dataset. Need not be defined."""
    CATEGORIES: List[str] = None
    """List of categories in the dataset. Need not be defined."""

    # ===================================== [ Registry stuff ] ======================================
    __data_module_registry = {}
    """Registry of data modules."""

    def __init_subclass__(cls, **kwargs):
        # check that the class attributes are defined
        super().__init_subclass__(**kwargs)
        assert cls.NUM_CLASSES is not None
        # rest don't need to be defined

        # get name and remove DataModule suffix
        name = cls.__name__
        # check if name matches XXXDataModule
        if not name.endswith("DataModule"):
            raise ValueError(
                f"Data module class name '{name}' does not end with 'DataModule'"
            )
        name = name[: -len("DataModule")]
        # check if name is already registered
        if name in cls.__data_module_registry:
            raise ValueError(f"Data module {name} already registered")
        # register the class in the registry
        cls.__data_module_registry[name] = cls

    @classmethod
    def registry(cls):
        """Returns the registry of data modules."""
        return cls.__data_module_registry

    # ===================================== [ Normal stuff ] ======================================
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.train_dataset = None
        self.eval_dataset = None

        mixup_alpha = config.get("mixup_alpha", 0.0)
        cutmix_alpha = config.get("cutmix_alpha", 0.0)
        p_gridified = config.get("p_gridified", 0.0)
        self.train_collate_fn = self.get_train_collate_fn(
            mixup_alpha, cutmix_alpha, p_gridified
        )

    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        shuffle = None if train_sampler is not None else True
        return data.DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def get_train_collate_fn(
        cls,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        p_gridified: float = 0.0,
    ):
        assert not (p_gridified and mixup_alpha), "For now, do not use both."

        collate_fn = None
        if p_gridified:
            gridify = SplitAndGrid(p_gridified, num_classes=cls.NUM_CLASSES)

            def collate_fn(batch):
                return gridify(*data.default_collate(batch))

            rank_zero_info(f"Gridify active for training with {p_gridified=}")

        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(cls.NUM_CLASSES, p=1.0, alpha=mixup_alpha)
            )
            rank_zero_info(f"Mixup active for training with {mixup_alpha=}")
        if cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(cls.NUM_CLASSES, p=1.0, alpha=cutmix_alpha)
            )
            rank_zero_info(f"Cutmix active for training with {cutmix_alpha=}")
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):  # noqa: F811
                return mixupcutmix(*data.default_collate(batch))

        return collate_fn

    def get_train_sampler(self):
        train_sampler = None

        # see https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/pytorch_lightning/trainer/connectors/data_connector.py#L336
        # and https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/lightning_lite/utilities/seed.py#L54
        seed = int(os.getenv("PL_GLOBAL_SEED", 0))
        ra_reps = self.config.get("ra_repetitions", None)
        if ra_reps is not None:
            rank_zero_info(f"Activating RASampler with {ra_reps=}")
            train_sampler = RASampler(
                self.train_dataset,
                shuffle=True,
                seed=seed,
                repetitions=ra_reps,
            )

        return train_sampler


class ImageNetDataModule(ClassificationDataModule):
    # from https://image-net.org/download.php
    NUM_CLASSES: int = 1000

    NUM_TRAIN_EXAMPLES: int = 1_281_167
    NUM_EVAL_EXAMPLES: int = 50_000

    CATEGORIES: List[str] = IMAGENET_CATEGORIES

    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_per_node = self.config.get("cache_dataset", None) == "shm"

    def prepare_data(self) -> None:
        cache_dataset = self.config.get("cache_dataset", None)
        if cache_dataset != "shm":
            return

        # print because we also want global non-zero rank's
        start = time.perf_counter()
        print("Caching dataset into SHM!...")
        from .caching import cache_tar_files_to_shm

        cache_tar_files_to_shm()
        end = time.perf_counter()
        print(f"Caching successful! Time taken {end - start:.2f}s")

    def setup(self, stage: str) -> None:
        # this way changes to the settings are reflected at function call time
        SHMTMPDIR = settings.SHMTMPDIR
        IMAGENET_PATH = settings.IMAGENET_PATH
        if stage == "fit":
            cache_dataset = self.config.get("cache_dataset", None)
            rank_zero_info("Setting up ImageNet train dataset...")
            start = time.perf_counter()
            train_root = os.path.join(
                SHMTMPDIR if cache_dataset == "shm" else IMAGENET_PATH,
                "train",
            )
            self.train_dataset = ImageFolder(
                root=train_root,
                transform=self.config["train_transform"],
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

            if cache_dataset == "onthefly":
                rank_zero_info("Trying to setup Bagua's cached dataset!")
                from .caching import CachedImageFolder

                self.train_dataset = CachedImageFolder(self.train_dataset)
                rank_zero_info("Successfully setup cached dataset!")

        start = time.perf_counter()
        rank_zero_info("Setting up ImageNet val dataset...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENET_PATH, "val"),
            transform=self.config["test_transform"],
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")


class CIFAR10DataModule(ClassificationDataModule):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    NUM_CLASSES: int = 10

    NUM_TRAIN_EXAMPLES: int = 50_000
    NUM_EVAL_EXAMPLES: int = 10_000

    CATEGORIES: List[str] = CIFAR10_CATEGORIES

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.DATA_ROOT
        if stage == "fit":
            self.train_dataset = CIFAR10(
                root=DATA_ROOT,
                train=True,
                transform=self.config["train_transform"],
                download=True,
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES

        self.eval_dataset = CIFAR10(
            root=DATA_ROOT,
            train=False,
            transform=self.config["test_transform"],
            download=True,
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES

class VOCDataModule(ClassificationDataModule):
    NUM_CLASSES: int = 20

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.VOC_PATH
        if stage == "fit":
            if self.config.get('train_split_portion', None) is not None:
                    entire_train_data = VOCDataset(
                        root=DATA_ROOT,
                        image_set='train',
                        download=False,
                        # transform=self.config["train_transform"], # Don't pass it here!
                        year='2012',
                        preload = self.config['preload'],
                        also_annotation=self.config['also_annotation'],
                    )
                    train_indices, eval_indices = get_random_cut(entire_train_data, self.config.get('train_split_portion'))
                    self.train_dataset = MySubset(entire_train_data,
                                indices=train_indices,
                                transform=self.config["train_transform"] 
                                )
                    self.eval_dataset = MySubset(entire_train_data,
                                indices=eval_indices,
                                transform=self.config["test_transform"] 
                                )
                    self.train_idx_trans = lambda idx: train_indices[idx]
                    self.eval_idx_trans = lambda idx: eval_indices[idx]
                    rank_zero_info(f'[Fit and Eval Setup] {len(self.train_dataset), len(self.eval_dataset)} for train and eval.')
                    rank_zero_info(f'Eval indices hash is: {hash(tuple(sorted(eval_indices)))}')
                    return
            else:
                self.train_dataset = VOCDataset(
                    root=DATA_ROOT,
                    image_set='train',
                    transform=self.config["train_transform"],
                    download=False,
                    year='2012',
                    preload = self.config['preload'],
                    also_annotation=self.config['also_annotation'],
                )

        if stage in ['fit', 'val'] and self.config.get('train_split_portion', None) is not None:
            rank_zero_info("Not Setting up anything as val split is part of fit and should be done in fit setup!")
            return

        eval_stage = stage
        if stage == 'fit':
            eval_stage = 'val'

        self.eval_dataset = VOCDataset(
            root=DATA_ROOT,
            image_set=eval_stage,
            transform=self.config["test_transform"],
            download=False,
            year='2012',
            preload=self.config['preload'],
            also_annotation=self.config['also_annotation'],
        )

class VOCDataset(torchvision.datasets.VOCDetection):
    def __init__(self, *args, preload=False, also_annotation=False, **kwargs):
        super(VOCDataset, self).__init__(*args, **kwargs)
        self.transforms = None

        self.target_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,
                'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        self.reverse_target_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7:
                       'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
                15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

        self.num_classes = 20

        if preload:
            self.preload = False
            self.load_data()
        self.preload = preload
        self.also_annotation = also_annotation
        assert self.transforms is None, f'Not considered as of now!'

    def load_data(self):
        rank_zero_info(f"Preloading all the data!")
        transform = self.transform
        target_transform = self.target_transform

        self.cached_images = []
        self.cached_targets = []
        self.transform = None
        self.target_transform = None
        for idx in range(len(self)):
            img, target = self[idx]
            self.cached_images[idx] = img
            self.cached_targets[idx] = target

        self.transform = transform
        self.target_transform = target_transform
        rank_zero_info(f"cached all the data successfully!")
        rank_zero_info(f"Putting back the transforms {self.transform=}, {self.target_transform=}")

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if self.preload:
            img = self.cached_images[index]
            target = self.cached_targets[index]
        else:
            img = Image.open(self.images[index]).convert("RGB")
            annotations = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

            objects = annotations['annotation']['object']
            target = torch.zeros(self.num_classes)
            object_names = [item['name'] for item in objects]
            for name in object_names:
                target[self.target_dict[name]] = 1

        if self.transform is not None:
            img = self.transform(img)
        
        if self.also_annotation:
            size = annotations['annotation']['size']
            width = int(size['width'])
            height = int(size['height'])
            wscale = 224 / width
            hscale = 224 / height

            object_bndboxes = [item['bndbox'] for item in objects]
            bbs = []
            for name, bndbox in zip(object_names, object_bndboxes):
                index = self.target_dict[name]
                xmin, xmax = int(bndbox['xmin']), int(bndbox['xmax'])
                ymin, ymax = int(bndbox['ymin']), int(bndbox['ymax'])

                new_xmin, new_xmax = int(min(max(xmin*wscale, 0), 223)), int(min(max(xmax*wscale, 0), 223))
                new_ymin, new_ymax = int(min(max(ymin*hscale, 0), 223)), int(min(max(ymax*hscale, 0), 223))

                bbs.append([index, new_xmin, new_ymin, new_xmax, new_ymax])
            return img, target, bbs
        else:
            return img, target
        
        # This applies transforms to both img and target (irrelevant for us!)
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

class MySubset(data.Subset):
    """
    Subset dataset with a few more things:
    - supporting a custom transform
    - delegate rest of attr./methods to internal dataset.

    Note: Mainly required for splitting and then using different transforms on
          created `Subset`s. (Otherwise, it's overwritten b/c internal is same.)
    Note: only for supervised data of form (x, y)
    """
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform
        if hasattr(dataset, "transform") and dataset.transform is not None:
            rank_zero_info(f"Internal dataset has transform will apply transform on top: {dataset}")

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __getattr__(self, item):
        if item in ['transform', 'target_transform']:
            return self.__dict__[item]
        # not found in attr so look in internal dataset
        return getattr(self.dataset, item)


# =================== Pneumonia Dataset and DataModule ===================
class PneumoniaDataset(data.Dataset):
    """RSNA Pneumonia dataset loader.

    Expects a CSV with at least columns:
      - patientId: DICOM filename stem (without .dcm)
      - Target: integer label (0/1)

    Images are read from: image_folder/{patientId}.dcm
    """

    def __init__(self, dataframe: pd.DataFrame, image_folder: str, transform=None, return_id: bool = False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        patient_id = str(row["patientId"])
        label = int(row["Target"])

        image_path = os.path.join(self.image_folder, f"{patient_id}.dcm")
        dicom = pydicom.dcmread(image_path)
        image = Image.fromarray(dicom.pixel_array).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        y = torch.tensor(label, dtype=torch.long)
        if self.return_id:
            return image, y, patient_id
        return image, y


class PneumoniaDataModule(ClassificationDataModule):
    """Lightning DataModule for RSNA Pneumonia classification.

    Configuration keys expected in `config`:
      - csv_path: path to CSV file with columns [patientId, Target]
      - image_folder: path to folder containing DICOM images named {patientId}.dcm

    Optional config keys:
      - splits_path: path to a pickle file containing a list of (train_idx, val_idx)
                     index arrays/iterables (like your KFold splits).
      - fold_index: int, which split to use from `splits_path` (default: 0)
      - sampling: bool, if True uses WeightedRandomSampler in train_dataloader
      - train_transform / test_transform: already used by base class and expected
      - batch_size / num_workers: already used by base class and expected

    Notes:
      - If `splits_path` is not provided, a deterministic 90/10 split is created.
      - `eval_dataset` is the fold's validation split.
    """

    NUM_CLASSES: int = 2

    def __init__(self, config):
        super().__init__(config)
        self.csv_path: str = self.config["csv_path"]
        self.image_folder: str = self.config["image_folder"]
        self.splits_path: Optional[str] = self.config.get("splits_path", None)
        self.fold_index: int = int(self.config.get("fold_index", 0))
        self.use_sampling: bool = bool(self.config.get("sampling", False))

        # cached to reuse between stages
        self._df: Optional[pd.DataFrame] = None
        self._splits: Optional[list] = None

    def _load_df_and_splits(self):
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)

        if self._splits is None:
            if self.splits_path is not None:
                with open(self.splits_path, "rb") as f:
                    self._splits = pickle.load(f)
            else:
                # deterministic 90/10 split
                all_idx = list(range(len(self._df)))
                train_idx, val_idx = get_random_cut(all_idx, 0.9)
                self._splits = [(train_idx, val_idx)]

        if not self._splits:
            raise ValueError("No splits available for PneumoniaDataModule")

        # clamp fold index
        self.fold_index = max(0, min(self.fold_index, len(self._splits) - 1))

    def setup(self, stage: str) -> None:
        self._load_df_and_splits()
        train_idx, val_idx = self._splits[self.fold_index]

        # `train_idx`/`val_idx` can be numpy arrays, lists, etc.
        train_df = self._df.iloc[list(train_idx)]
        val_df = self._df.iloc[list(val_idx)]

        return_id = bool(self.config.get("return_id", False))

        if stage == "fit":
            self.train_dataset = PneumoniaDataset(
                train_df,
                self.image_folder,
                transform=self.config["train_transform"],
                return_id=return_id,
            )
            self.eval_dataset = PneumoniaDataset(
                val_df,
                self.image_folder,
                transform=self.config["test_transform"],
                return_id=return_id,
            )
        elif stage in ["val", "test"]:
            # In Lightning, validation can be called without fit in some flows.
            self.eval_dataset = PneumoniaDataset(
                val_df,
                self.image_folder,
                transform=self.config["test_transform"],
                return_id=return_id,
            )

    def train_dataloader(self):
        # Optional weighted sampling for class imbalance
        if self.use_sampling:
            # Compute sample weights from labels in the underlying dataframe
            # (works because PneumoniaDataset stores a reset-indexed dataframe)
            labels = self.train_dataset.dataframe["Target"].astype(int)
            class_counts = labels.value_counts().to_dict()
            class_weights = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
            sample_weights = labels.map(class_weights).astype(float).values.tolist()
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
            return data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.train_collate_fn,
                pin_memory=True,
            )

        return super().train_dataloader()

#
# =================== VinBigData Multi-Label Dataset and DataModule ===================
class VinBigXrayMultiLabelDataset(data.Dataset):
    """VinBigData chest X-ray multi-label dataset.

    Expects a CSV where the first column is an image identifier/filename and the
    remaining columns are multi-label targets (0/1) for each abnormality.

    Common formats:
      - column name `image_id` with files stored as `<image_id>.png`
      - or a column that already contains a filename with extension

    Images are read from: image_folder/<filename>

    Returns:
      image: transformed image tensor
      target: float tensor of shape [NUM_CLASSES] with 0/1 values
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_folder: str,
        transform=None,
        image_col: Optional[str] = None,
        num_classes: int = 14,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform
        self.num_classes = int(num_classes)

        # Choose the column that identifies the image file
        if image_col is not None:
            self.image_col = image_col
        elif "image_id" in self.dataframe.columns:
            self.image_col = "image_id"
        elif "image" in self.dataframe.columns:
            self.image_col = "image"
        elif "filename" in self.dataframe.columns:
            self.image_col = "filename"
        else:
            # fall back to first column
            self.image_col = str(self.dataframe.columns[0])

        # All other columns are assumed to be labels
        self.label_cols = [c for c in self.dataframe.columns if c != self.image_col]
        if len(self.label_cols) == 0:
            raise ValueError(
                "VinBigXrayMultiLabelDataset: No label columns found. "
                "Expected one image-id column and remaining label columns."
            )

        # If the CSV has more/less labels than expected, we still allow it
        # but clamp/pad at runtime.

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[int(idx)]
        img_id = str(row[self.image_col])

        # Determine filename
        if img_id.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            filename = img_id
        else:
            filename = f"{img_id}.png"

        image_path = os.path.join(self.image_folder, filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Multi-label target vector
        y = row[self.label_cols].to_numpy(dtype=np.float32)

        # Clamp/pad to num_classes (defensive)
        if y.shape[0] > self.num_classes:
            y = y[: self.num_classes]
        elif y.shape[0] < self.num_classes:
            y2 = np.zeros((self.num_classes,), dtype=np.float32)
            y2[: y.shape[0]] = y
            y = y2

        target = torch.from_numpy(y)
        return image, target


class VinBigXrayDataModule(ClassificationDataModule):
    """Lightning DataModule for VinBigData multi-label classification.

    Required config keys:
      - csv_path: CSV path
      - image_folder: folder containing PNG/JPG images

    Optional config keys:
      - splits_path: pickle file containing list of (train_idx, val_idx)
      - fold_index: which split to use
      - sampling: bool, if True uses multi-label weighted sampling (see old script)
      - image_col: optional column name for the image id/filename
      - num_classes: number of labels (default 14)
      - train_transform / test_transform: torchvision transforms
      - batch_size / num_workers

    Notes:
      - Weighted sampling follows your old script:
            label_counts = sum over label columns
            class_weights = 1/(counts+eps), clipped
            sample_weight = labels dot class_weights, normalized
    """

    NUM_CLASSES: int = 14

    # Optional human-readable class names (from your older script)
    CATEGORIES: List[str] = [
        "Aortic Enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Consolidation",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Nodule/Mass",
        "Other lesion",
        "Pleural Effusion",
        "Pleural Thickening",
        "Pneumothorax",
        "Pulmonary Fibrosis",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.csv_path: str = self.config["csv_path"]
        self.image_folder: str = self.config["image_folder"]
        self.splits_path: Optional[str] = self.config.get("splits_path", None)
        self.fold_index: int = int(self.config.get("fold_index", 0))
        self.use_sampling: bool = bool(self.config.get("sampling", False))
        self.image_col: Optional[str] = self.config.get("image_col", None)
        self.num_classes: int = int(self.config.get("num_classes", self.NUM_CLASSES))

        # cached to reuse between stages
        self._df: Optional[pd.DataFrame] = None
        self._splits: Optional[list] = None

    def _load_df_and_splits(self):
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)

        if self._splits is None:
            if self.splits_path is not None:
                with open(self.splits_path, "rb") as f:
                    self._splits = pickle.load(f)
            else:
                # deterministic 90/10 split
                all_idx = list(range(len(self._df)))
                train_idx, val_idx = get_random_cut(all_idx, 0.9)
                self._splits = [(train_idx, val_idx)]

        if not self._splits:
            raise ValueError("No splits available for VinBigXrayDataModule")

        self.fold_index = max(0, min(self.fold_index, len(self._splits) - 1))

    def setup(self, stage: str) -> None:
        self._load_df_and_splits()
        train_idx, val_idx = self._splits[self.fold_index]

        train_df = self._df.iloc[list(train_idx)]
        val_df = self._df.iloc[list(val_idx)]

        if stage == "fit":
            self.train_dataset = VinBigXrayMultiLabelDataset(
                train_df,
                self.image_folder,
                transform=self.config["train_transform"],
                image_col=self.image_col,
                num_classes=self.num_classes,
            )
            self.eval_dataset = VinBigXrayMultiLabelDataset(
                val_df,
                self.image_folder,
                transform=self.config["test_transform"],
                image_col=self.image_col,
                num_classes=self.num_classes,
            )
        elif stage in ["val", "test"]:
            self.eval_dataset = VinBigXrayMultiLabelDataset(
                val_df,
                self.image_folder,
                transform=self.config["test_transform"],
                image_col=self.image_col,
                num_classes=self.num_classes,
            )

    def train_dataloader(self):
        if self.use_sampling:
            # Compute per-sample weights from multi-label targets (like your old script)
            df = self.train_dataset.dataframe

            # identify label columns
            image_col = self.train_dataset.image_col
            label_cols = self.train_dataset.label_cols

            labels = df[label_cols].to_numpy(dtype=np.float32)
            label_counts = labels.sum(axis=0)  # [C]
            class_weights = 1.0 / (label_counts + 1e-6)
            # mimic your clip to avoid extreme weights
            class_weights = np.clip(class_weights, a_min=1 / 3, a_max=3.0)

            sample_weights = labels.dot(class_weights)  # [N]
            # normalize for numerical stability (not required but matches your script)
            if sample_weights.max() > 0:
                sample_weights = sample_weights / sample_weights.max()

            seed = int(os.getenv("PL_GLOBAL_SEED", 0))
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            )

            return data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.train_collate_fn,
                pin_memory=True,
            )

        return super().train_dataloader()


class CC3MDataModule(ClassificationDataModule):
    NUM_CLASSES: int = -1 # How to handle this?

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.CC3M_PATH
        
        collator = CustomDataCollatorImg()
        cc3m_obj = CC3MImg()

        if stage == "fit":
            split_path = "training"
            tar_name = "{00000..00331}.tar"
            data_shard = os.path.join(DATA_ROOT, split_path, tar_name)
            self.train_dataset = cc3m_obj.get_wds_dataset(
                data_shard, 
                self.config["train_transform"], 
                self.batch_size, 
                collator=collator)

        split_path = "validation"
        tar_name = "{00000..00001}.tar"
        data_shard = os.path.join(DATA_ROOT, split_path, tar_name)
        self.eval_dataset = cc3m_obj.get_wds_dataset(
                data_shard, 
                self.config["test_transform"], 
                self.batch_size, 
                collator=collator)
    
    # # Following loaders are not the default but adapated as per the cc3m.py code from Sukrut
    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        # shuffle = None if train_sampler is not None else True
        shuffle = False
        return data.DataLoader(
            self.train_dataset,
            None, 
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )