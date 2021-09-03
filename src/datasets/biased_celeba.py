import torch
import numpy as np

from torch.utils.data import Dataset
from os.path import join
from PIL import Image
from torchvision import transforms as tt

class BiasedCelebA(Dataset):
    """Version of the CelebA dataset with a custom data
       partition and labels
    """
    all_attributes = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split(" ")
    def __init__(self, path, labels_path, split, transforms):
        """Constructior

        Args:
            path (str): path where images are located
            labels_path (str): path were label partition is located
            split (str): train, val or test
            transforms (torchvision.transforms.Transform): transforms to apply
        """
        self.path = path
        self.transforms = transforms
        self.x = []
        self.y = []
        splits = ["train", "val", "test"]
        current_split = splits.index(split)
        with open(labels_path, 'r') as infile:
            for i, line in enumerate(infile.readlines()):
                line = line.replace('\n', '')
                if i == 0:
                    self.attributes = line.split(',')[2:]
                else:
                    img, _split, *labels = line.split(',')
                    if int(_split) == current_split:
                        self.x.append(img)
                        self.y.append(list(map(int, labels)))
        self.y = np.stack(self.y, 0)
        self.y = (self.y == 1).astype(int)
        self.class_count = self.y.sum(0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img = Image.open(join(self.path, 'img_align_celeba', 'img_align_celeba', self.x[item]))
        return self.transforms(img), torch.LongTensor(self.y[item])

