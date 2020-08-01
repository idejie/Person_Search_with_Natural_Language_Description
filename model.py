import torch
import torch.nn as nn
from torchvision.models import vgg16


class Attention(nn.Module):
    def __init__(self, conf):
        super(Attention, self).__init__()
        # generate visual units
        self.visual_unit = nn.Sequential(
            nn.Linear(4096, conf.embedding_size),
            nn.BatchNorm1d(conf.embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(conf.embedding_size, conf.output_size)
        )

        # word-level gate
        self.word_gate = nn.Sequential(
            nn.Linear(conf.rnn_hidden_size, 1),
            nn.Sigmoid()
        )

        # Unit-level attention
        self.unit_attention = nn.Sequential(
            nn.Linear(conf.rnn_hidden_size, conf.output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, images_feats, rnn_out):
        # print('image_feats attn', images_feats.shape)
        images_out = self.visual_unit(images_feats)
        images_out_batch = images_out.repeat(rnn_out.size(1), 1, 1).transpose(0, 1)
        # print('images_out_batch attn', images_out_batch.shape)  # (batch_size,seq_len,512)
        word_gate_out = self.word_gate(rnn_out)
        # print('word_gate_out attn', word_gate_out.shape)  # (batch_size,seq_len,1)
        unit_attention_out = self.unit_attention(rnn_out)
        # print('unit_attention_out attn', unit_attention_out.shape)  # (batch_size,seq_len,512)
        images_out_batch = images_out_batch.transpose(1, 2)
        # print('images_out_batch attn', images_out_batch.shape)  # (batch_size,512,seq_len)
        dotproct = torch.bmm(unit_attention_out, images_out_batch)
        # dotproct = dotproct.view(len(dotproct), -1, 1)
        # print('dotproct attn', dotproct.shape)  # (batch_size,seq_len,seq_len)
        affinity = torch.matmul(dotproct, word_gate_out)
        # print('affinity attn', affinity.shape)  # (batch_size,seq_len,seq_len)
        return affinity


class Language_Subnet(nn.Module):
    def __init__(self, conf):
        super(Language_Subnet, self).__init__()
        self.word_emb = nn.Linear(conf.vocab_size, conf.embedding_size)
        self.image_emb = nn.Sequential(
            nn.Linear(4096, conf.embedding_size),  # vis-fc1
            nn.BatchNorm1d(conf.embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(conf.embedding_size, conf.embedding_size)  # vis-fc2
        )
        self.rnn = nn.LSTM(input_size=conf.embedding_size * 2,
                           hidden_size=conf.rnn_hidden_size,
                           num_layers=conf.rnn_layers,
                           dropout=conf.rnn_dropout
                           )
        self.conf = conf
        self.attention = Attention(conf)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_feats, captions):
        x_v = self.image_emb(image_feats)
        # print('x_v', x_v.shape)

        # print('captions', captions.shape)
        x_w_t = self.word_emb(captions)
        # print('x_w_t', x_w_t.shape)
        x_v = x_v.repeat(x_w_t.size(1), 1, 1).transpose(0, 1)
        # print('x_v', x_v.shape)
        x_emb = torch.cat([x_w_t, x_v], dim=2)

        # print('x_emb', x_emb.shape)
        hidden_state_0 = torch.zeros((self.conf.rnn_layers, x_emb.size(1), self.conf.rnn_hidden_size))
        cell_state_0 = torch.zeros((self.conf.rnn_layers, x_emb.size(1), self.conf.rnn_hidden_size))
        if self.conf.gpu_id != -1:
            hidden_state_0 = hidden_state_0.cuda()
            cell_state_0 = cell_state_0.cuda()
        rnn_out, (hidden_states, cell_states) = self.rnn(x_emb, (hidden_state_0, cell_state_0))
        # print('rnn_out', rnn_out.shape)
        attn_out = self.attention(image_feats, rnn_out)
        # print('attn_out', attn_out.shape)
        attn_out = torch.sum(attn_out, dim=1)
        out = self.sigmoid(attn_out)
        # out = torch.mean(out, dim=1)
        return out


class GNA_RNN(nn.Module):
    def __init__(self, conf):
        super(GNA_RNN, self).__init__()
        # visual sub-network
        self.cnn = vgg16(pretrained=True)
        self.cnn.classifier = self.cnn.classifier[:-1]
        self.language_subnet = Language_Subnet(conf)

    def forward(self, images, captions):
        image_feats = self.cnn(images)
        out = self.language_subnet(image_feats, captions)
        return out
