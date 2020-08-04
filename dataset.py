import os
import random

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
        self.positive_samples = conf.positive_samples
        self.negative_samples = conf.negative_samples
        self.n_original_captions = conf.n_original_captions
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        """ get an item of dataset by index

        Examples:
            If `positive_samples = 2` and `negative_samples = 1`, the length of dataset will be `(negative_samples+ positive_samples) = 3`.
            Then the index `0, 1, 2, 3, 4, 5, 6, 7` will point to a same image `I`.
            And the caption of index 0 and 3 will be the first original caption of image `I`, index 1 will be the  the
            second original caption.
            And the other indexes will sample a different caption of a different image.

            So `index % (negative_samples+ positive_samples) <= 1` will be positive sample.
               ``

        Args:
            index: the index of the dataset

        Returns:
            item: an item of dataset
        """

        pos_img_index = index // ((self.negative_samples + self.positive_samples) * self.n_original_captions)
        pos_data = self.dataset[pos_img_index]
        image_path = os.path.join(self.config.images_dir, pos_data['file_path'])

        image = Image.open(image_path)
        #  resize image to 256x256
        image = self.transform(image)

        # sample caption
        t_index = index % (self.negative_samples + self.positive_samples)
        if t_index < self.n_original_captions:
            # postive
            # transform captions to (1,max_length) one-hot vector
            cap_index = pos_data['index_captions'][t_index]
            label = 1
        else:
            # negative samples
            label = 0
            # sample a different image
            neg_img_index = random.randint(0, len(self.dataset) - 1)
            while neg_img_index == pos_img_index:
                neg_img_index = random.randint(0, len(self.dataset) - 1)
            neg_data = self.dataset[neg_img_index]
            # sample a caption
            neg_cap_index = random.randint(0, self.n_original_captions - 1)
            cap_index = neg_data['index_captions'][neg_cap_index]
            assert neg_data != pos_data, 'the negative sample should different with positive sample'
        caption = torch.zeros((self.config.max_length, self.config.vocab_size))
        for i, cap_i in enumerate(cap_index):
            if i < self.config.max_length:
                caption[i][cap_i] = 1
        return image, caption, label

    def __len__(self):
        """get the length of the dataset

        Returns:
            the length of the dataset
        """
        return len(self.dataset) * (self.negative_samples + self.positive_samples) * self.n_original_captions
