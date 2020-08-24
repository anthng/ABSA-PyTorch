# -*- coding: utf-8 -*-
# file: cnn.py
# author: anng <thienan99dt@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.squeeze_embedding import SqueezeEmbedding

"""
#twitter
>> test_acc: 0.7298, test_f1: 0.7104

"""

class CNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CNN, self).__init__()
        """
            - in_channels (int) – Number of channels in the input image

            - out_channels (int) – Number of channels produced by the convolution

            - kSize (int or tuple) – Size of the convolving kernel

            - dropout: should be 0.2

            - lr: should be 1e-4
        """
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)

        self.squeeze_embedding = SqueezeEmbedding()

        filters = 100
        #kernel size
        kSize = [3,4,5]

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=(kSize[0], self.opt.embed_dim), bias=True)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=(kSize[1], self.opt.embed_dim), bias=True)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=(kSize[2], self.opt.embed_dim), bias=True)

        ### Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        feature = len(kSize) * filters

        self.dropout = nn.Dropout(opt.dropout)
        #self.dropout = nn.Dropout(0.2)
        #self.softmax = nn.Softmax()
        self.dense = nn.Linear(feature, self.opt.polarities_dim, bias=True)


    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)

        embed = self.squeeze_embedding(x, x_len)
        embed = self.dropout(embed)
        embed = embed.unsqueeze(1)
        #print("Embed size",embed.size())
        out0 = self.tanh(self.conv0(embed)).squeeze(3)
        out1 = self.tanh(self.conv1(embed)).squeeze(3)
        out2 = self.tanh(self.conv2(embed)).squeeze(3)
        #print("out0 squeeze 3",out0.size())
        out0 = F.max_pool1d(out0, out0.size(2)).squeeze(2)
        out1 = F.max_pool1d(out1, out1.size(2)).squeeze(2)
        out2 = F.max_pool1d(out2, out2.size(2)).squeeze(2)
        #print("out0 squeeze 2",out0.size())

        x = torch.cat([out0,out1,out2], dim = 1)
        x = self.dropout(x)

        out = self.dense(x)
        #print(out.size())
        return out
