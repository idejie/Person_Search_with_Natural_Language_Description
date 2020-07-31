from torch.utils.data.dataset import Dataset


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
        self.dataset = dataset
        pass

    def __getitem__(self, index):
        """ get an item of dataset by index

        Args:
            index: the index of the dataset

        Returns:
            item: an item of dataset
        """
        #  resize image to 256x256
        pass

    def __len__(self):
        """get the length of the dataset

        Returns:
            the length of the dataset
        """
        return len(self.dataset)
