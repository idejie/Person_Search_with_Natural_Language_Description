import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CUHK_PEDES
from model import GNA_RNN
from utils.config import Config
from utils.preprocess import *


class Model(object):
    """the class of model

    Attributes:

    """

    def __init__(self, conf):
        if type(conf.gpu_id) == int and torch.cuda.is_available():
            self.device = torch.device('cuda:' + str(conf.gpu_id))
        if conf.backend == 'cudnn' and torch.cuda.is_available():
            cudnn.benchmark = True
        conf.logger.info(self.device)
        self.logger = conf.logger
        self.vocab_dir = conf.vocab_dir
        self.data_dir = conf.data_dir
        self.raw_data = conf.raw_data
        self.word_count_threshold = conf.word_count_threshold
        self.conf = conf
        # load data
        if conf.action != 'process':
            train_set, valid_set, test_set, vocab = self.load_data()
            conf.vocab_size = vocab['UNK'] + 1
            self.train_set = CUHK_PEDES(conf, train_set, is_train=True)
            self.valid_set = CUHK_PEDES(conf, valid_set)
            self.test_set = CUHK_PEDES(conf, test_set)
            self.train_loader = DataLoader(self.train_set, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                           shuffle=True)
            self.valid_loader = DataLoader(self.valid_set, batch_size=conf.batch_size, num_workers=conf.num_workers)
            self.test_loader = DataLoader(self.test_set, batch_size=1)

            # init network
            self.net = GNA_RNN(conf)
            self.criterion = nn.BCELoss()
            self.optimizer = Adam(params=self.net.parameters(), lr=1e-5)
            self.lr_scheduler = None

    def load_data(self):
        train_set_path = os.path.join(self.data_dir, 'train_set.json')
        with open(train_set_path, 'r', encoding='utf8') as f:
            train_set = json.load(f)
        valid_set_path = os.path.join(self.data_dir, 'valid_set.json')
        with open(valid_set_path, 'r', encoding='utf8') as f:
            valid_set = json.load(f)
        test_set_path = os.path.join(self.data_dir, 'test_set.json')
        with open(test_set_path, 'r', encoding='utf8') as f:
            test_set = json.load(f)
        w2i_path = os.path.join(self.vocab_dir, 'w2i.json')
        with open(w2i_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        return train_set, valid_set, test_set, vocab

    def process(self):
        # load images list
        raw_data_path = os.path.join(self.data_dir, self.raw_data)
        with open(raw_data_path, 'r', encoding='utf8') as f:
            images_info = json.load(f)

        #  tokenize captions
        # images_tokenized = tokenize(images)

        # create the vocab
        vocab, images_info = build_vocab(images_info, self.word_count_threshold)
        i2w = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
        w2i = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

        # save vocab-index map
        if not os.path.exists(self.vocab_dir):
            os.mkdir(self.vocab_dir)
        i2w_path = os.path.join(self.vocab_dir, 'i2w.json')
        with open(i2w_path, 'w', encoding='utf8') as f:
            json.dump(i2w, f, indent=2)
        w2i_path = os.path.join(self.vocab_dir, 'w2i.json')
        with open(w2i_path, 'w', encoding='utf8') as f:
            json.dump(w2i, f, indent=2)

        # encode captions
        train_set, val_set, test_set = encode_captions(images_info, w2i)

        # save the splitting dataset
        train_set_path = os.path.join(self.data_dir, 'train_set.json')
        with open(train_set_path, 'w', encoding='utf8') as f:
            json.dump(train_set, f)
        valid_set_path = os.path.join(self.data_dir, 'valid_set.json')
        with open(valid_set_path, 'w', encoding='utf8') as f:
            json.dump(val_set, f)
        test_set_path = os.path.join(self.data_dir, 'test_set.json')
        with open(test_set_path, 'w', encoding='utf8') as f:
            json.dump(test_set, f)

    def train(self):
        for e in range(self.conf.epochs):
            for b, (images, captions, labels) in enumerate(self.train_loader):
                self.net.train()
                self.optimizer.zero_grad()
                labels = labels.unsqueeze(-1).float()
                if self.conf.gpu_id != -1:
                    self.net.cuda()
                    images = images.cuda()
                    captions = captions.cuda()
                    labels = labels.cuda()
                out = self.net(images, captions)
                loss = self.criterion(out, labels)
                self.logger.info(
                    f'Epoch {e}/{self.conf.epochs} Batch {b}/{len(self.train_loader)}, Loss:{loss.item():.4f}')
                loss.backward()
                self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

    def test(self):
        self.net.eval()
        for b_cap, (correct_images, query_captions, _) in enumerate(self.test_loader):
            if self.conf.gpu_id != -1:
                self.net.cuda()
                correct_images = correct_images.cuda()
                query_captions = query_captions.cuda()
            for b_img, (database_images,_,_) in enumerate(self.test_loader):
                if self.conf.gpu_id !=-1:
                    database_images = database_images.cuda()
                out = self.net(database_images, query_captions)

            print(out)
            loss = self.criterion(out, labels)

    def eval(self):
        self.net.eval()
        for b, (images, captions, labels) in enumerate(self.valid_loader):
            pass

    def web(self):
        pass


def main():
    conf = Config()
    model = Model(conf)
    if conf.action == 'process':
        conf.logger.info('start to pre-process data')
        model.process()
    elif conf.action == 'train':
        conf.logger.info('start to train')
        model.train()
    elif conf.action == 'test':
        conf.logger.info('start to test')
        model.test()
    elif conf.action == 'web':
        conf.logger.info('start to run a web')
        model.web()
    else:
        raise KeyError(f'No support fot this action: {conf.action}')


if __name__ == '__main__':
    main()
