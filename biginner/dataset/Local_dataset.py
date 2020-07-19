import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class JungleDataset(Dataset):
    def __init__(self, config, split='train', transform=True, is_train=True):
        self.root = config.data_root
        self.transform = transform
        self.data_num = []
        self.is_train = is_train
        self.split = split
        self.rgb2gray = np.zeros(256 ** 3)
        for i, cm in enumerate(config.jungle_colormap):
            self.rgb2gray[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        data_list_file = os.path.join(self.root, self.split, '{0}_list.txt'.format(self.split))
        self.data_num = [id_.strip() for id_ in open(data_list_file)]

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, i_):
        i_num = self.data_num[i_]
        img = Image.open(os.path.join(self.root, self.split, 'image', i_num + 'img.png'))
        label = Image.open(os.path.join(self.root, self.split, 'label', i_num + 'label.png')).convert('RGB')

        if self.transform is True:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.is_train:
                img = T.ToTensor()(img)
                img = normalize(img)
                label = self.image2label(label)
                label = torch.from_numpy(label)

            else:
                img = T.ToTensor()(img)
                img = normalize(img)
                label = self.image2label(label)
                label = torch.from_numpy(label)
        return img, label

    def image2label(self, im):
        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.rgb2gray[idx], dtype='int64')