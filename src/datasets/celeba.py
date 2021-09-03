from torch.utils.data import Dataset
from os.path import join
import numpy as np
from PIL import Image
from torchvision import transforms as tt
import torch

class CelebA(Dataset):
    def __init__(self, path, split, transforms):
        self.path = path
        self.transforms = transforms
        dataset = {}
        with open(join(path, 'list_attr_celeba.csv'), 'r') as infile:
            for i, line in enumerate(infile.readlines()):
                if i == 0:
                    self.attributes = line.split(',')[1:]
                else:
                    img, *labels = line.split(',')
                    dataset[img] = list(map(int, labels))
        self.split_names = []
        self.x = []
        self.y = []
        splits = ["train", "val", "test"]
        current_split = splits.index(split)
        with open(join(path, 'list_eval_partition.csv'), 'r') as infile: 
            for i, line in enumerate(infile.readlines()):
                if i == 0:
                    continue
                img_name, _split = line.split(',')
                if int(_split) == current_split:
                    self.x.append(img_name)
                    self.y.append(dataset[img_name])
        self.y = np.stack(self.y, 0)
        self.y = (self.y == 1).astype(int)
        self.class_count = self.y.sum(0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img = Image.open(join(self.path, 'img_align_celeba', 'img_align_celeba', self.x[item]))
        return self.transforms(img), torch.LongTensor(self.y[item])



