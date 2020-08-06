import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import CUHK_PEDES
from model import GNA_RNN
from utils.config import Config
from utils.preprocess import *

config = Config()
if config.action in ['train', 'test']:
    if config.amp:
        from torch.cuda.amp import autocast
    if config.parallel:
        from torch import distributed

        # init distributed backend
        distributed.init_process_group(backend='nccl')
        # configuration for different GPUs


class Model(object):
    """the class of model

    Attributes:

    """

    def __init__(self, conf):
        conf.logger.info(f'CUDA is available? {torch.cuda.is_available()}')
        if type(conf.gpu_id) == int and torch.cuda.is_available():
            device = f'Single-GPU:{conf.gpu_id}'
            self.device = torch.device(f'cuda:{conf.gpu_id}')
        elif type(conf.gpu_id) == list and len(conf.gpu_id) > 0 and torch.cuda.device_count() > 1:
            device = f'Multi-GPU:{conf.gpu_id}'
            local_rank = distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')
            self.device = device
        conf.logger.info(device)
        if conf.backend == 'cudnn' and torch.cuda.is_available():
            cudnn.benchmark = True

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
            self.valid_query_set = CUHK_PEDES(conf, valid_set, query_or_db='query')
            self.valid_db_set = CUHK_PEDES(conf, valid_set, query_or_db='db')
            self.test_query_set = CUHK_PEDES(conf, test_set, query_or_db='query')
            self.test_db_set = CUHK_PEDES(conf, test_set, query_or_db='db')
            if conf.parallel:
                self.train_loader = DataLoader(self.train_set, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                               sampler=DistributedSampler(self.train_set))
                self.valid_query_loader = DataLoader(self.valid_query_set, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers,
                                                     sampler=DistributedSampler(self.valid_query_set))
                self.valid_db_loader = DataLoader(self.valid_db_set, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers,
                                                  sampler=DistributedSampler(self.valid_db_set))
                self.test_query_loader = DataLoader(self.test_query_set, batch_size=conf.batch_size,
                                                    num_workers=conf.num_workers,
                                                    sampler=DistributedSampler(self.test_query_set))
                self.test_db_loader = DataLoader(self.test_db_set, batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers,
                                                 sampler=DistributedSampler(self.test_db_set))
            else:
                self.train_loader = DataLoader(self.train_set, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                               shuffle=True)
                self.valid_query_loader = DataLoader(self.valid_query_set, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers)
                self.valid_db_loader = DataLoader(self.valid_db_set, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers)
                self.test_query_loader = DataLoader(self.test_query_set, batch_size=conf.batch_size,
                                                    num_workers=conf.num_workers)
                self.test_db_loader = DataLoader(self.test_db_set, batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers)

            # init network
            self.net = GNA_RNN(conf)
            if conf.parallel:
                self.net.to(self.device)
                self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank],
                                                                     output_device=local_rank)
                self.optimizer = Adam([
                    {'params': self.net.module.language_subnet.parameters(), 'lr': conf.language_lr},
                    {'params': self.net.module.cnn.parameters(), 'lr': conf.cnn_lr}
                ])
            else:
                self.optimizer = Adam([
                    {'params': self.net.language_subnet.parameters(), 'lr': conf.language_lr},
                    {'params': self.net.cnn.parameters(), 'lr': conf.cnn_lr}
                ])

            self.criterion = nn.BCEWithLogitsLoss()
            all_steps = self.conf.epochs * len(self.train_loader)
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                  T_max=all_steps / 2 if conf.parallel else all_steps)

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
        # train stage
        if self.conf.amp:
            scaler = torch.cuda.amp.GradScaler()
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
                if self.conf.amp:
                    with autocast():
                        out = self.net(images, captions)
                        loss = self.criterion(out, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    out = self.net(images, captions)
                    loss = self.criterion(out, labels)
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.logger.info(
                    f'Epoch {e}/{self.conf.epochs} Batch {b}/{len(self.train_loader)}, Loss:{loss.item():.4f}')
                if (b + 1) % self.conf.eval_interval == 0:
                    self.eval()

            self.save_checkpoint(e)
            if (e + 1) % self.conf.test_interval == 0:
                self.test()

    def test(self):
        # test stage
        self.net.eval()
        n_query = len(self.test_query_loader.dataset)
        n_database = len(self.test_db_loader.dataset)
        out_matrix = np.zeros((n_query, n_database))
        labels_matrix = np.zeros((n_query, n_database))
        eval_bar = tqdm(total=len(self.test_query_loader) * len(self.test_db_loader.dataset.dataset),
                        desc='Test Stage')
        with torch.no_grad():
            # images: for d in db
            for d, (images, indexes_d) in enumerate(self.test_db_loader):
                if self.conf.gpu_id != -1:
                    images = images.cuda()
                if self.conf.amp:
                    with autocast():
                        images_feats_out = self.net(images)
                else:
                    images_feats_out = self.net(images)
                # caption: for q in  query
                for q, (captions, indexes_q) in enumerate(self.test_query_loader):
                    if self.conf.gpu_id != -1:
                        captions = captions.cuda()
                    for image_out, index_d in zip(images_feats_out, indexes_d):
                        image_out_repeat = image_out.repeat(len(captions), 1)
                        index_d_repeat = index_d.repeat(len(captions), 1)
                        # print(image_out_repeat.shape,index_d_repeat.shape)
                        if self.conf.amp:
                            with autocast():
                                outs = self.net(image_out_repeat, captions, query=True)
                        else:
                            outs = self.net(image_out_repeat, captions, query=True)
                        eval_bar.update(1)
                        out_matrix[indexes_q, index_d_repeat] = outs.squeeze(1).cpu().detach().numpy()
                        labels = (index_d_repeat == indexes_q) + 0
                        labels_matrix[indexes_q, index_d_repeat] = labels.numpy()
        out_matrix = torch.from_numpy(out_matrix)
        labels_matrix = torch.from_numpy(labels_matrix)
        eval_bar.close()
        if self.conf.gpu_id != -1:
            out_matrix = out_matrix.cuda()
            labels_matrix = labels_matrix.cuda()
        if self.conf.amp:
            with autocast():
                loss = self.criterion(out_matrix, labels_matrix)
        else:
            loss = self.criterion(out_matrix, labels_matrix)
        self.logger.info(f'Test average loss: {loss.item():.4f}')
        metrics = self.calculate_metrics(out_matrix, labels_matrix)
        return loss, metrics

    def eval(self):
        # eval stage
        self.net.eval()
        n_query = len(self.valid_query_loader.dataset)
        n_database = len(self.valid_db_loader.dataset)
        out_matrix = np.zeros((n_query, n_database))
        labels_matrix = np.zeros((n_query, n_database))
        eval_bar = tqdm(total=len(self.valid_query_loader) * len(self.valid_db_loader.dataset.dataset),
                        desc='Eval Stage')
        with torch.no_grad():
            # images: for d in db
            for d, (images, indexes_d) in enumerate(self.valid_db_loader):
                if self.conf.gpu_id != -1:
                    images = images.cuda()
                if self.conf.amp:
                    with autocast():
                        images_feats_out = self.net(images)
                else:
                    images_feats_out = self.net(images)
                # caption: for q in  query
                for q, (captions, indexes_q) in enumerate(self.valid_query_loader):
                    if self.conf.gpu_id != -1:
                        captions = captions.cuda()
                    # repeat image for batch input
                    for image_out, index_d in zip(images_feats_out, indexes_d):
                        image_out_repeat = image_out.repeat(len(captions), 1)
                        index_d_repeat = index_d.repeat(len(captions), 1)
                        if self.conf.amp:
                            with autocast():
                                outs = self.net(image_out_repeat, captions, query=True)
                        else:
                            outs = self.net(image_out_repeat, captions, query=True)
                        eval_bar.update(1)
                        out_matrix[indexes_q, index_d_repeat] = outs.squeeze(1).cpu().detach().numpy()
                        labels = (index_d_repeat == indexes_q) + 0
                        labels_matrix[indexes_q, index_d_repeat] = labels.numpy()
        out_matrix = torch.from_numpy(out_matrix)
        labels_matrix = torch.from_numpy(labels_matrix)
        eval_bar.close()
        if self.conf.gpu_id != -1:
            out_matrix = out_matrix.cuda()
            labels_matrix = labels_matrix.cuda()
        if self.conf.amp:
            with autocast():
                loss = self.criterion(out_matrix, labels_matrix)
        else:
            loss = self.criterion(out_matrix, labels_matrix)
        self.logger.info(f'Eval average loss: {loss.item():.4f}')
        metrics = self.calculate_metrics(out_matrix, labels_matrix)
        return loss, metrics

    def query_transform(self, query):
        w2i_file = os.path.join(self.vocab_dir, 'w2i.json')
        with open(w2i_file, 'r', encoding='utf8') as f:
            w2i = json.load(f)
        tokens = ''.join(c for c in query.strip().lower() if c not in string.punctuation)
        tokens = tokens.split()
        query_vector = []
        for token in tokens:
            query_vector.append(w2i.get(token, w2i['UNK']))
        return query_vector

    def web(self):
        from flask import Flask
        from flask import render_template
        app = Flask("Person Search with Natural Language Description", static_folder='data/CUHK-PEDES/imgs',
                    template_folder='web')
        self.load_checkpoint('checkpoints', 'epoch_8.cpt')

        @app.route('/')
        def home():
            return 'Person Search with Natural Language Description'

        def search(query_vector, k=20):
            top = []
            caption = torch.Tensor(query_vector)
            if self.conf.gpu_id != -1:
                caption = caption.cuda()
            n_database = len(self.test_db_loader.dataset)
            out_matrix = np.zeros(n_database)
            for images, indexes_q in self.test_db_loader:
                if self.conf.gpu_id != -1:
                    images = images.cuda()
                if self.conf.amp:
                    with autocast():
                        images_feats_out = self.net(images)
                else:
                    images_feats_out = self.net(images)
                captions = caption.repeat(len(images_feats_out), 1, 1)
                if self.conf.amp:
                    with autocast():
                        out = self.net(images_feats_out, captions, query=True)
                else:
                    out = self.net(images_feats_out, captions, query=True)
                out_matrix[indexes_q] = out.squeeze(1).cpu().detach().numpy()
            out_matrix = torch.tensor(out_matrix)
            _, top_k_indexes = out_matrix.topk(k, sorted=True)
            for index in top_k_indexes:
                data = self.test_db_loader.dataset.dataset[index]
                path = data['file_path']
                top.append(path)
            return top

        @app.route('/q')
        @app.route('/q/<q>')
        def query(q=None):
            if q is None:
                return render_template('web/query.html')
            else:
                result = {'q': q}
                query_vector = self.query_transform(q)
                result['vector'] = query_vector
                top_20 = search(query_vector, k=20)
                result['top'] = top_20
                return render_template('web/result.html', result=result)

        app.run(debug=self.conf.web_debug)

    def calculate_metrics(self, out_matrix, labels_matrix):
        sorted_indexes = out_matrix.argsort(dim=1)
        r = []
        for k in self.conf.top_k:
            n_corrects = 0
            indexes_k = sorted_indexes[:, :k]
            for index_k, labels in zip(indexes_k, labels_matrix):
                ret = labels[index_k]
                if sum(ret) > 0:
                    n_corrects += 1
            acc = n_corrects / len(labels_matrix)
            self.logger.info(f'top-{k} acc: {acc * 100.0:.2f}%')
            r.append(acc)
        return r

    def save_checkpoint(self, e, checkpoints_dir=None):
        file_name = f'epoch_{e}.cpt'
        if checkpoints_dir is None:
            checkpoints_dir = self.conf.checkpoints_dir
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        file_path = os.path.join(checkpoints_dir, file_name)
        content = {
            'model': self.net.module.state_dict() if config.parallel else self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': e,
            'parallel': self.conf.parallel

        }
        torch.save(content, file_path)
        self.logger.info(f'saved checkpoints: {file_name}')

    def load_checkpoint(self, checkpoints_dir=None, file_name=None):
        if checkpoints_dir is None or file_name is None:
            raise FileNotFoundError('please set checkpoint directory and filename')
        file_path = os.path.join(checkpoints_dir, file_name)
        content = torch.load(file_path)
        if content.get('parallel', True):
            net = torch.nn.parallel.DistributedDataParallel(self.net)
            net.load_state_dict(content['model'])
        else:
            self.net.load_state_dict(content['model'])
        self.logger.info(f'loaded checkpoints from `{file_path}`')


def main():
    d = config.__dict__.copy()
    d.pop('logger')
    j = json.dumps(d, indent=2)
    config.logger.info('\n' + j)
    model = Model(config)

    if config.action == 'process':
        config.logger.info('start to pre-process data')
        model.process()
    elif config.action == 'train':
        config.logger.info('start to train')
        model.train()
    elif config.action == 'test':
        config.logger.info('start to test')
        model.test()
    elif config.action == 'web':
        config.logger.info('start to run a web')
        model.web()
    else:
        raise KeyError(f'No support fot this action: {config.action}')


if __name__ == '__main__':
    main()
