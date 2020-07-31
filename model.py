import torch
import torch.nn as nn
from torchvision.models import vgg16


class Attention(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(Attention, self).__init__()
        # generate visual units
        self.visual_unit = nn.Sequential(
            nn.Linear(4096, embedding_size),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, output_size)
        )

        # word-level gate
        self.word_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Unit-level attention
        self.unit_attention = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )

    def forward(self, images, captions):
        images_out = self.visual_unit(images)
        word_gate_out = self.word_gate(captions)
        unit_attention_out = self.unit_attention(captions)
        dotproct = torch.bmm(unit_attention_out, images_out)
        dotproct = dotproct.view(len(dotproct), -1, 1)
        affinity = torch.matmul(dotproct, word_gate_out)

        return affinity


class GNA_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNA_RNN, self).__init__()
        # visual sub-network
        self.vgg16 = vgg16(pretrained=True)
        self.vgg16.classifier = self.vgg16.classifier[:-1]
        self.visual_unit = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.Linear(output_size, output_size)
        )

        # language sub-network
        self.word_fc1 = nn.Linear(1, hidden_size)
        self.word_lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, images, captions):
        vgg_feats = self.vgg16(images)
        x_v = self.visual_unit(vgg_feats)

        x_t_w = self.word_fc1(captions)
        gna_rnn_out = torch.cat([x_t_w, x_v], dim=1)
        word_lstm_out = self.word_lstm(gna_rnn_out)
        out = None
        # todo: add attention
        return out
