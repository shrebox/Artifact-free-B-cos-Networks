import random
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

"""
Imgaug-free augmentations that are similar to the original imgaug pipeline.

Original heavy imgaug block:
    iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2)))
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0)))
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255)))

We replicate that with:
- Coarse salt/pepper via coarse grid sampling and upsampling (very imgaug-like)
- Gaussian blur via torchvision.transforms.GaussianBlur with sigma in (0,2)
- Additive Gaussian noise in pixel domain (0..255), scale uniform in (0, 0.04*255)
"""


# -------------------------
# Imgaug-like primitives
# -------------------------

class Sometimes:
    """Apply transform with probability p (like imgaug.Sometimes)."""
    def __init__(self, p: float, transform):
        self.p = float(p)
        self.transform = transform

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self.transform(img)
        return img


class CoarseSaltAndPepper:
    """
    Imgaug-like CoarseSaltAndPepper:
    - Choose coarse cell size as % of image dims: size_percent in [0.1,0.2]
    - Sample salt/pepper decisions on the coarse grid
    - Upsample masks by repeating cells back to full resolution
    """
    def __init__(
        self,
        p_salt: float = 0.01,
        p_pepper: float = 0.01,
        size_percent_range: Tuple[float, float] = (0.1, 0.2),
    ):
        self.p_salt = float(p_salt)
        self.p_pepper = float(p_pepper)
        self.size_percent_range = (float(size_percent_range[0]), float(size_percent_range[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]

        sp = random.uniform(*self.size_percent_range)

        # cell size in pixels ~ size_percent * image_dim
        cell_h = max(1, int(round(h * sp)))
        cell_w = max(1, int(round(w * sp)))

        # grid resolution
        gh = int(np.ceil(h / cell_h))
        gw = int(np.ceil(w / cell_w))

        rnd = np.random.rand(gh, gw)
        salt = rnd < self.p_salt
        pepper = (rnd >= self.p_salt) & (rnd < self.p_salt + self.p_pepper)

        # upsample by repeating coarse grid
        salt_full = np.repeat(np.repeat(salt, cell_h, axis=0), cell_w, axis=1)[:h, :w]
        pep_full  = np.repeat(np.repeat(pepper, cell_h, axis=0), cell_w, axis=1)[:h, :w]

        # Apply to image
        if arr.ndim == 2:
            # grayscale
            arr[salt_full] = 255
            arr[pep_full] = 0
        else:
            # RGB: set all channels for selected pixels
            arr[salt_full, :] = 255
            arr[pep_full, :] = 0

        return Image.fromarray(arr)


class GaussianBlurImgaugLike:
    """
    Imgaug-like GaussianBlur(sigma=(0,2)).
    torchvision GaussianBlur needs kernel_size. We'll use a kernel size based on sigma.
    """
    def __init__(self, sigma_range: Tuple[float, float] = (0.0, 2.0)):
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(*self.sigma_range)

        # imgaug allows sigma=0 => no-op; torchvision blur with sigma=0 can be weird, so handle explicitly
        if sigma <= 1e-8:
            return img

        # choose kernel size similar to common Gaussian practice: k ~ 6*sigma + 1, forced odd, min 3
        k = int(round(6 * sigma + 1))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        blur = transforms.GaussianBlur(kernel_size=k, sigma=(sigma, sigma))
        return blur(img)


class AdditiveGaussianNoiseImgaugLike:
    """
    Imgaug-like AdditiveGaussianNoise(scale=(0, 0.04*255)).
    In imgaug, 'scale' is the stddev of the noise in pixel space.
    We'll sample sigma uniformly in that range and add N(0, sigma^2).
    """
    def __init__(self, sigma_range: Tuple[float, float] = (0.0, 0.04 * 255.0)):
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(*self.sigma_range)
        if sigma <= 1e-8:
            return img

        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr)


class HeavyImageAugmentationSupport:
    """
    Replacement for the old imgaug pipeline.
    """
    def __init__(self):
        self.pipeline = transforms.Compose([
            Sometimes(0.1, CoarseSaltAndPepper(p_salt=0.01, p_pepper=0.01, size_percent_range=(0.1, 0.2))),
            Sometimes(0.5, GaussianBlurImgaugLike(sigma_range=(0.0, 2.0))),
            Sometimes(0.5, AdditiveGaussianNoiseImgaugLike(sigma_range=(0.0, 0.04 * 255.0))),
        ])

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.pipeline(img)


# -------------------------
# Public API (same functions)
# -------------------------

def get_no_augmentations_no_resize():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_no_augmentations_resize():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_light_augmentations_resize():
    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.Lambda(lambda img: TF.affine(
            img, angle=0,
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),
            scale=1.0, shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img, angle=0,
            translate=(0, 0),
            scale=1.0 / (2 ** random.gauss(0, 0.1)),
            shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=random.gauss(0, 5),
            translate=(0, 0),
            scale=1.0,
            shear=random.gauss(0, 2.5)
        )),

        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.20))),
        transforms.ToTensor(),
    ])


def get_light_augmentations_no_resize():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(
            img, angle=0,
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),
            scale=1.0, shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img, angle=0,
            translate=(0, 0),
            scale=1.0 / (2 ** random.gauss(0, 0.1)),
            shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=random.gauss(0, 5),
            translate=(0, 0),
            scale=1.0,
            shear=random.gauss(0, 2.5)
        )),

        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.20))),
        transforms.ToTensor(),
    ])


def get_heavy_augmentations_no_rotation_no_resize():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),
            scale=1.0,
            shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,
            translate=(0, 0),
            scale=1.0 / (2 ** random.gauss(0, 0.15)),
            shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=random.gauss(0, 6),
            translate=(0, 0),
            scale=1.0,
            shear=random.gauss(0, 4)
        )),

        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.25))),

        # imgaug-like noise/blur block
        HeavyImageAugmentationSupport(),

        transforms.ToTensor(),
    ])


def get_heavy_augmentations_no_rotation_resize():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),
            scale=1.0,
            shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,
            translate=(0, 0),
            scale=1.0 / (2 ** random.gauss(0, 0.15)),
            shear=0
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=random.gauss(0, 6),
            translate=(0, 0),
            scale=1.0,
            shear=random.gauss(0, 4)
        )),

        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.25))),

        HeavyImageAugmentationSupport(),

        transforms.ToTensor(),
    ])
