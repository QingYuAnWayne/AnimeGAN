import torch as t
from torch.utils.data import Dataset
import os
from torchvision import transforms as T
from PIL import Image
from config import DefaultConfig

opt = DefaultConfig()


class AnimeImgs(Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        super(AnimeImgs, self).__init__()
        imgs = [os.path.join(root, img) for img in os.listdir(path=root)]

        def get_index(x):
            x = x.split('.')[-2].split('/')[-1]
            return x

        self.imgs = sorted(imgs, key=lambda x: get_index(x))
        imgs_num = len(imgs)

        if transforms:
            self.transform = T.Compose([
                T.Resize(opt.image_size),
                T.CenterCrop(opt.image_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        """
        返回一张图片的数据
        1000.jpg 返回 label = 1000
        """
        img_path = self.imgs[index]
        label = index
        data = Image.open(img_path, "r")
        data = self.transforms(data)
        return data, label
