import torch
import torch.nn as nn
from torchvision.models import vgg16


class Attention(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(Attention, self).__init__()
        # generate visual units
        self.visual_unit = nn.Sequential(
            nn.Linear(4096, embedding_size),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, output_size)
        )

        # word-level gate
        self.word_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        # Unit-level attention
        self.attention_fc = nn.Linear(hidden_size, output_size)
        self.unit_attention = nn.Softmax()

    def forward(self, images, captions):
        images_out = self.visual_unit(images)

        # return affinity


class GNA_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GNA_RNN, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.vgg16.classifier = self.vgg16.classifier[:-1]
        self.vis_fc1 = nn.Linear(4096, hidden_size)
        self.vis_fc2 = nn.Linear(hidden_size, 512)

        self.word_fc1 = nn.Linear(1, hidden_size)

        self.word_lstm = Word_LSTM(input_size, hidden_size)

    def forward(self, images, captions):
        vgg_feats = self.vgg16(images)
        vgg_out_1 = self.vis_fc1(vgg_feats)
        x_v = self.vis_fc2(vgg_out_1)

        x_t_w = self.word_fc1(captions)
        gna_rnn_out = torch.cat([x_t_w, x_v], dim=1)
        word_lstm_out = self.word_lstm(gna_rnn_out)



class Word_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        return self.lstm(x)


GNA_RNN()
