import os
import numpy as np

from torchvision.datasets import DatasetFolder

from .celeba import CelebA
from .biased_celeba import BiasedCelebA
from PIL import Image
from torchvision import transforms as tt



def get_dataset(dataset_path, split, exp_dict):
    flip = "flip" in exp_dict.get("transforms", [])
    labels_path = os.path.join(dataset_path, 'celeba', exp_dict[f"labels_path"])
    dataset_path = os.path.join(dataset_path, exp_dict[f"dataset_{split}"])
    
    if exp_dict["dataset"] == "celeba":
        t = tt.Compose([tt.Resize(max(exp_dict["height"], exp_dict["width"]), Image.BICUBIC),
                        tt.CenterCrop((exp_dict["height"], exp_dict["width"])),
                        tt.RandomHorizontalFlip() if split == 'train' and flip else lambda x: x,
                        tt.ToTensor(),
                        tt.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        dataset =  CelebA(dataset_path,
                      split,
                      transforms=t)

    elif exp_dict["dataset"] == "biased_celeba":
        t = tt.Compose([tt.CenterCrop(exp_dict["crop"]) if "crop" in exp_dict else lambda x: x,
                        tt.Resize((exp_dict["height"], exp_dict["width"]), Image.BICUBIC),
                        tt.RandomHorizontalFlip() if split == 'train' and flip else lambda x: x,
                        tt.ToTensor(),
                        tt.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        # print(dataset_path)
        # print(labels_path)
        dataset = BiasedCelebA(dataset_path,
                      labels_path,
                      split,
                      transforms=t)

    return dataset
