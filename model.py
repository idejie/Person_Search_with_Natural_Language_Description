import torch
import torch.nn as nn
from torchvision.models import vgg16


class Attention(nn.Module):
    def __init__(self, conf):
        super(Attention, self).__init__()
        # generate visual units
        self.visual_unit = nn.Sequential(
            nn.Linear(4096, conf.embedding_size),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.Linear(conf.embedding_size, conf.output_size)
        )

        # word-level gate
        self.word_gate = nn.Sequential(
            nn.Linear(conf.hidden_size, 1),
            nn.Sigmoid()
        )

        # Unit-level attention
        self.unit_attention = nn.Sequential(
            nn.Linear(conf.hidden_size, conf.output_size),
            nn.Softmax()
        )

    def forward(self, images_feats, rnn_out):
        images_out = self.visual_unit(images_feats)
        word_gate_out = self.word_gate(rnn_out)
        unit_attention_out = self.unit_attention(rnn_out)
        dotproct = torch.bmm(unit_attention_out, images_out)
        dotproct = dotproct.view(len(dotproct), -1, 1)
        affinity = torch.matmul(dotproct, word_gate_out)

        return affinity


class Language_Subnet(nn.Module):
    def __init__(self, conf):
        super(Language_Subnet, self).__init__()
        self.word_emb = nn.Embedding(conf.vocab_size, conf.embedding_size)
        self.image_emb = nn.Sequential(
            nn.Linear(4096, conf.embedding_size),  # vis-fc1
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.Linear(conf.embedding_size, conf.embedding_size)  # vis-fc2
        )
        self.rnn = nn.LSTM(input_size=conf.embedding_size * 2,
                           hidden_size=conf.rnn_hidden_size,
                           num_layers=conf.rnn_layers,
                           dropout=conf.rnn_dropout
                           )
        self.attention = Attention(conf)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_feats, captions):
        x_w_t = self.word_emb(captions)
        x_v = self.image_emb(image_feats)
        x_emb = torch.cat((x_w_t, x_v), dim=1)
        rnn_out, hidden_state = self.rnn(x_emb)
        attn_out = self.attention(image_feats, rnn_out)
        out = self.sigmoid(attn_out)
        return out


class GNA_RNN(nn.Module):
    def __init__(self, conf):
        super(GNA_RNN, self).__init__()
        # visual sub-network
        self.cnn = vgg16(pretrained=True)
        self.cnn.classifier = self.vgg16.classifier[:-1]
        self.language_subnet = Language_Subnet(conf)

    def forward(self, images, captions):
        image_feats = self.vgg16(images)

        out = self.language_subnet(image_feats, captions)
        return out
