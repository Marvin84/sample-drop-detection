#@author: Santiago Pascual - UPC, Barcelona, Spain
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from config import *
from IPython import embed


class RNNSampleLoss(nn.Module):
    def __init__(self, num_inputs=512,
                 rnn_size=1024,
                 rnn_layers=1, rnn_dropout=0):
        super().__init__()
        self.num_inputs = num_inputs
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(num_inputs, 512, 5,
                      padding=5//2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 512, 5,
                      padding=5//2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 512, 1)
        )

        self.rnn = nn.LSTM(512, rnn_size,
                           num_layers=rnn_layers,
                           dropout=rnn_dropout,
                           batch_first=True)
        self.out_mlp = nn.Sequential(
            nn.Linear( 2*rnn_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.att = MultiHeadedAttention(8, 1024)

        self.out_mlp_att = nn.Sequential(
                       nn.Linear(rnn_size, 512),
                       nn.BatchNorm1d(512),
                       nn.ReLU(inplace=True),
                       nn.Linear(512, 1),
                       nn.Sigmoid()
        )

    def forward(self, Xhyp, Xref):
        Xref = Xref.transpose(1, 2)
        Xhyp = Xhyp.transpose(1, 2)
        href = self.cnn_frontend(Xref)
        hhyp = self.cnn_frontend(Xhyp)
        href = href.transpose(1, 2)
        hhyp = hhyp.transpose(1, 2)
        href, _ = self.rnn(href)
        hhyp, _ = self.rnn(hhyp)
        if not config_isAttention:
            hcat = torch.cat((href[:, -1, :], hhyp[:, -1, :]), dim=1)
            return self.out_mlp(hcat)
        else:
            Y = self.att(hhyp, hhyp, href)
            return self.out_mlp_att(Y[:,-1,:])
            #return self.out_mlp_att(torch.mean(Y, dim=1))



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #print(scores.shape)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Figure 2"

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


#
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, config_featureMapValues, (1, 9))
#         self.conv1_bn = nn.BatchNorm2d(config_featureMapValues)
#         self.conv2 = nn.Conv2d(config_featureMapValues, config_featureMapValues, (1, 21))
#         self.conv2_bn = nn.BatchNorm2d(config_featureMapValues)
#         self.conv3 = nn.Conv2d(config_featureMapValues, config_featureMapValues, (1, 11))
#         self.conv3_bn = nn.BatchNorm2d(config_featureMapValues)
#         self.conv4 = nn.Conv2d(config_featureMapValues, config_featureMapValues, (1, 5))
#         self.conv4_bn = nn.BatchNorm2d(config_featureMapValues)
#         self.flattenOutput = nn.Linear(6*config_featureMapValues*config_fqValues, 1)
#         self.outActivation = nn.Sigmoid()
#
#     def forward(self, x):
#         x = F.relu(self.conv1_bn(self.conv1(x)))
#         x = F.relu(self.conv2_bn(self.conv2(x)))
#         x = F.relu(self.conv3_bn(self.conv3(x)))
#         x = F.relu(self.conv4_bn(self.conv4(x)))
#         #x = F.relu(self.conv5_bn(self.conv5(x)))
#         x = x.view(x.size(0), -1)
#         x = self.flattenOutput(x)
#         x = self.outActivation(x)
#
#         return x





