import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class CUHK_PEDES(Dataset):
    """ the class for CUHK_PEDES dataset

    Attributes:

    """

    def __init__(self, conf, dataset, is_train=False, query_or_db='query'):
        """ init CUHK_PEDES class

        Args:
            conf:
            dataset:
        """
        self.split = dataset[0]["split"]
        self.is_train = is_train
        self.query_or_db = query_or_db
        self.dataset = dataset
        if not is_train:
            self.dataset = self.dataset[:1000]
        self.config = conf
        self.positive_samples = conf.positive_samples
        self.negative_samples = conf.negative_samples
        self.n_original_captions = conf.n_original_captions
        self.transform = transforms.Compose([
            transforms.Resize(conf.image_size),
            transforms.ToTensor()
        ])
        if is_train:
            conf.logger.info(f'init {self.split}  dataset, length:{len(self)}')
        else:
            conf.logger.info(f'init {self.split} {query_or_db}  dataset, length:{len(self)}')

    def __getitem__(self, index):
        """ get an item of dataset by index

        Args:
            index: the index of the dataset

        Returns:
            item: an item of dataset
        """

        if self.is_train:

            pos_img_index = index // ((self.negative_samples + self.positive_samples) * self.n_original_captions)
            pos_data = self.dataset[pos_img_index]
            image_path = os.path.join(self.config.images_dir, pos_data['file_path'])
            # sample caption
            t_index = index % ((self.negative_samples + self.positive_samples) * self.n_original_captions)

            if t_index < self.positive_samples * self.n_original_captions:
                # positive
                # transform captions to (1,max_length) one-hot vector
                cap_index = t_index % self.n_original_captions
                cap_index = pos_data['index_captions'][cap_index]
                label = 1
            else:
                # negative samples
                label = 0
                # sample a different image
                neg_img_index = random.randint(0, len(self.dataset) - 1)
                while self.dataset[neg_img_index]['id'] == pos_data['id']:
                    neg_img_index = random.randint(0, len(self.dataset) - 1)
                neg_data = self.dataset[neg_img_index]
                # sample a negative caption
                neg_cap_index = random.randint(0, self.n_original_captions - 1)
                # transform captions to (1,max_length) one-hot vector
                cap_index = neg_data['index_captions'][neg_cap_index]
                assert neg_data != pos_data, 'the negative sample should different with positive sample'
            image = Image.open(image_path)
            # resize image to 256x256
            image = self.transform(image)
            # caption
            caption = np.zeros(self.config.max_length)
            for i, cap_i in enumerate(cap_index):
                if i < self.config.max_length:
                    caption[i] = cap_i
            caption = torch.LongTensor(caption)
            return image, caption, label
        else:
            if self.query_or_db == 'query':
                data_index = index // self.n_original_captions
                data = self.dataset[data_index]
                cap_index = index % self.n_original_captions
                cap_index = data['index_captions'][cap_index]
                caption = np.zeros(self.config.max_length)
                for i, cap_i in enumerate(cap_index):
                    if i < self.config.max_length:
                        caption[i] = cap_i
                caption = torch.LongTensor(caption)
                p_id = int(data['id'])
                return caption, index, p_id
            elif self.query_or_db == 'db':
                data = self.dataset[index]
                image_path = os.path.join(self.config.images_dir, data['file_path'])
                image = Image.open(image_path)
                # resize image to 256x256
                image = self.transform(image)
                p_id = int(data['id'])
                return image, index, p_id

    def __len__(self):
        """get the length of the dataset

        Returns:
            the length of the dataset
        """
        if self.is_train:
            return len(self.dataset) * (self.negative_samples + self.positive_samples) * self.n_original_captions
        else:
            if self.query_or_db == 'query':
                return len(self.dataset) * self.n_original_captions
            elif self.query_or_db == 'db':
                return len(self.dataset)
