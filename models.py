# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import torch.nn as nn
from transformers import AutoModel


class MLP(nn.Module):
    def __init__(self, num_labels, dropout, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, num_labels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        # x = nn.functional.relu(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class RobertaMLP(nn.Module):
    def __init__(self, num_labels, bert_path, drop_out=0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained(bert_path)

        self.mlp = MLP(
            num_labels=num_labels, 
            dropout=drop_out, 
            hidden_size=self.model.config.hidden_size
        )
        # self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask).last_hidden_state.mean(dim=-2)
        x = self.mlp(x)

        return x
