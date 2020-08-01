import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class CUHK_PEDES(Dataset):
    """ the class for CUHK_PEDES dataset

    Attributes:

    """

    def __init__(self, conf, dataset):
        """ init CUHK_PEDES class

        Args:
            conf:
            dataset:
        """
        self.split = dataset[0]["split"]
        conf.logger.info(f'init {self.split} dataset')
        self.dataset = dataset
        self.config = conf
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        """ get an item of dataset by index

        Args:
            index: the index of the dataset

        Returns:
            item: an item of dataset
        """
        cap_index = index % 2
        index = index // 2

        data = self.dataset[index]
        image_path = os.path.join(self.config.images_dir, data['file_path'])

        image = Image.open(image_path)
        #  resize image to 256x256
        image = self.transform(image)

        # transform captions to (1,max_length) one-hot vector
        cap_index = data['index_captions'][cap_index]

        caption = torch.zeros((self.config.max_length, self.config.vocab_size))
        for i, cap_i in enumerate(cap_index):
            if i < self.config.max_length:
                caption[i][cap_i] = 1

        label = 1

        return image, caption, label

    def __len__(self):
        """get the length of the dataset

        Returns:
            the length of the dataset
        """
        return len(self.dataset) * 2
