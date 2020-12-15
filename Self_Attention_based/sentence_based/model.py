# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU

from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class SelfAttention(nn.Module):
    def __init__(self, key_dim, value_dim, device):
        super(SelfAttention, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, queries, keys, values, mask):
        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim) # (batch_size, max_len, key_dim)
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim) # (batch_size, max_len, value_dim)
        
        queries.data.masked_fill_(key_mask.byte(), 0.)
        keys.data.masked_fill_(key_mask.byte(), 0.)
        values.data.masked_fill_(value_mask.byte(), 0.)

        max_len = keys.shape[1]
        weights = torch.bmm(queries, keys.permute(0,2,1)) # (batch_size, max_len, max_len)
        weight_mask = torch.where(weights == 0., torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.byte(), -float('inf'))
        for i in range(max_len):
            weights[:,i,i] = -float('inf')

        weights = self.softmax(weights)
        weight_mask = torch.where(torch.isnan(weights), torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.byte(), 0.)
        
        return torch.bmm(weights, values) # (batch_size, max_len, value_dim)


class Attention(nn.Module):
    def __init__(self, key_dim, value_dim, device):
        super(Attention, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, keys, values, mask):
        query = query.unsqueeze(1) # (batch_size, 1, key_dim)

        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim) # (batch_size, max_len, key_dim)
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim) # (batch_size, max_len, value_dim)
        
        keys.data.masked_fill_(key_mask.byte(), 0.)
        values.data.masked_fill_(value_mask.byte(), 0.)

        max_len = keys.shape[1]
        weights = torch.bmm(query, keys.permute(0,2,1)) # (batch_size, 1, max_len)
        weight_mask = torch.where(weights == 0., torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.byte(), -float('inf'))

        weights = self.softmax(weights)
        weight_mask = torch.where(torch.isnan(weights), torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.byte(), 0.) # (batch_size, 1, max_len)
        
        return torch.bmm(weights, values).squeeze(1) # (batch_size, value_dim)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class MemexQA(nn.Module):
    def __init__(self, input_dim, img_dim, hidden_dim, key_dim, value_dim, num_label=2, num_head=2, num_layer=1, mode='one-shot', device=torch.device("cpu")):
        super(MemexQA, self).__init__()
        self.device = device
        self.num_label = num_label
        self.num_head = num_head
        self.num_layer = num_layer
        self.mode = mode
        
        # img_emb
        # self.img_emb = nn.Linear(img_dim, hidden_dim)
        self.img_emb = nn.Sequential(
            nn.Linear(img_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.input_layer_norm = nn.LayerNorm(hidden_dim)

        # que_emb
        self.que_emb = nn.Linear(input_dim, key_dim*self.num_head)

        # ans_emb
        if self.mode == 'att-concat-one-shot':
            self.ans_emb = nn.Linear(input_dim, key_dim*self.num_head)
        else:
            self.ans_emb = nn.Linear(input_dim, value_dim*self.num_head)

        # self attention
        for i in range(self.num_head):
            setattr(self, 'self_sentence_query'+str(i+1), nn.Linear(input_dim, key_dim))
            setattr(self, 'self_sentence_key'+str(i+1), nn.Linear(input_dim, key_dim))
            setattr(self, 'self_sentence_value'+str(i+1), nn.Linear(input_dim, value_dim))
            setattr(self, 'self_image_query'+str(i+1), nn.Linear(hidden_dim, key_dim))
            setattr(self, 'self_image_key'+str(i+1), nn.Linear(hidden_dim, key_dim))
            setattr(self, 'self_image_value'+str(i+1), nn.Linear(hidden_dim, value_dim))
        self.self_attention = SelfAttention(key_dim, value_dim, self.device)
        self.layer_norm = nn.LayerNorm(value_dim*self.num_head)

        # More layers
        dim = self.num_head * value_dim
        for nl in range(self.num_layer-1):
            for i in range(self.num_head):
                setattr(self, 'self_query_{}_{}'.format(nl+1, i+1), nn.Linear(dim, key_dim))
                setattr(self, 'self_key_{}_{}'.format(nl+1, i+1), nn.Linear(dim, key_dim))
                setattr(self, 'self_value_{}_{}'.format(nl+1, i+1), nn.Linear(dim, value_dim))

        # question attention
        self.key_proj   = nn.Linear(value_dim*self.num_head, key_dim*self.num_head)
        self.value_proj = nn.Linear(value_dim*self.num_head, value_dim*self.num_head)
        self.attention  = Attention(key_dim*self.num_head, value_dim*self.num_head, self.device)

        # Prediction
        if self.mode == 'one-shot':
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head, self.num_label)
        elif self.mode == 'select-one': # NOT GREAT
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head, 1)
        elif self.mode == 'att-concat-one-shot':
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head*2, self.num_label)
        else:
             raise NotImplementedError("Not implemented!")

        # criterion
        # self.criterion = nn.CrossEntropyLoss()
        if 'one-shot' in self.mode:
            self.criterion = LabelSmoothingLoss(2, 0.1)
        elif 'select-one' in self.mode:
            self.criterion = LabelSmoothingLoss(4, 0.1)
        self.softmax = nn.Softmax(dim=-1)

        # Additional Techniques
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, question, answer, text, text_lengths, images, image_lengths, label):
        question = self.que_emb(question) # (bs*4, key_dim)
        answer   = self.ans_emb(answer) # (bs*4, value_dim)
        images   = self.img_emb(images) # (bs*4, ..., hidden_dim)

        batch_size = question.shape[0]
        text, images = self.input_layer_norm(text), self.input_layer_norm(images)

        queries, keys, values = [], [], []
        for i in range(self.num_head):
            sentence_quries = getattr(self,'self_sentence_query'+str(i+1))(text)
            image_quries    = getattr(self,'self_image_query'+str(i+1))(images)
            queries.append(torch.cat([sentence_quries, image_quries], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            sentence_keys = getattr(self,'self_sentence_key'+str(i+1))(text)
            image_keys    = getattr(self,'self_image_key'+str(i+1))(images)
            keys.append(torch.cat([sentence_keys, image_keys], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            sentence_values = getattr(self,'self_sentence_value'+str(i+1))(text)
            image_values    = getattr(self,'self_image_value'+str(i+1))(images)
            values.append(torch.cat([sentence_values, image_values], 1)) # (num_head, bs*4, text_len+image_len, value_dim)
        
        # mask for self_attention
        max_text_len, max_image_len = text.shape[1], images.shape[1]
        mask = np.ones((batch_size, max_text_len+max_image_len))
        for id, (tl, il) in enumerate(zip(text_lengths, image_lengths)):
            mask[id][:tl] = np.zeros_like(mask[id][:tl])
            mask[id][max_text_len:max_text_len+il] = np.zeros_like(mask[id][max_text_len:max_text_len+il])
        mask = torch.Tensor(mask).to(self.device)

        album_feat = []
        for i in range(self.num_head):
            self_att = self.self_attention(queries[i], keys[i], values[i], mask)
            album_feat.append(self_att)
        album_feat = torch.cat(album_feat, -1) # (bs*4, ..., value_dim*num_head)
        if self.num_layer > 1:
            album_feat = self.dropout(album_feat)
            album_feat = self.layer_norm(album_feat)

        # More self-attention-layers
        for nl in range(self.num_layer-1):
            queries, keys, values = [], [], []
            for i in range(self.num_head):
                query = getattr(self, 'self_query_{}_{}'.format(nl+1, i+1))(album_feat)
                queries.append(query)

                key = getattr(self, 'self_key_{}_{}'.format(nl+1, i+1))(album_feat)
                keys.append(key)

                value = getattr(self, 'self_value_{}_{}'.format(nl+1, i+1))(album_feat)
                values.append(value)

            new_album_feat = []
            for i in range(self.num_head):
                self_att = self.self_attention(queries[i], keys[i], values[i], mask)
                new_album_feat.append(self_att)
            new_album_feat = torch.cat(new_album_feat, -1) # (bs*4, text_len+image_len, value_dim*num_head)

            if nl < self.num_layer-2:
                album_feat = self.dropout(album_feat)
                album_feat = self.layer_norm(album_feat)

            album_feat = new_album_feat
            

        # Attention
        album_keys   = self.key_proj(album_feat) # (bs*4, ..., key_dim)
        album_values = self.value_proj(album_feat) # (bs*4, ..., value_dim)

        # Projection
        atts_question = self.attention(question, album_keys, album_values, mask) # (bs*4, value_dim)
        if self.mode == 'one-shot':
            outputs = torch.cat([question, answer, atts_question, answer*atts_question], 1) # (bs*4, value_dim*3+key_dim)
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            prediction = self.softmax(outputs)[:, 0] # (bs*4)
            loss = self.criterion(outputs, label)
        elif self.mode == 'select-one':
            outputs = torch.cat([question, answer, atts_question, answer*atts_question], 1) # (bs*4, value_dim*3+key_dim)
            outputs = self.answer_proj(outputs) # (bs*4, 1)
            prediction = self.softmax(outputs.view(-1, 4)) # (bs, 4)
            loss = self.criterion(prediction, torch.LongTensor(np.zeros(prediction.shape[0])).to(self.device))
        elif self.mode == 'att-concat-one-shot':
            atts_answer = self.attention(answer, album_keys, album_values, mask) # (bs*4, value_dim)
            outputs = torch.cat([question, answer, atts_answer, atts_question, atts_question*atts_question], 1) # (bs*4, value_dim*3+key_dim*2)
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            prediction = self.softmax(outputs)[:, 0] # (bs*4)
            loss = self.criterion(outputs, label)
        else:
            raise NotImplementedError("Not implemented!")

        return prediction, loss


class NaiveQA(nn.Module):
    def __init__(self, input_dim, img_dim, hidden_dim, key_dim, value_dim, num_label=2, num_head=2, num_layer=1, mode='one-shot', device=torch.device("cpu")):
        super(NaiveQA, self).__init__()
        self.device = device
        self.num_label = num_label
        self.num_head = num_head
        self.num_layer = num_layer
        self.mode = mode

        # que_emb
        self.que_emb = nn.Linear(input_dim, value_dim*self.num_head)

        # ans_emb
        self.ans_emb = nn.Linear(input_dim, value_dim*self.num_head)

        # Prediction
        if self.mode == 'one-shot':
            self.answer_proj = nn.Linear(value_dim*self.num_head*3, self.num_label)
        elif self.mode == 'select-one': # NOT GREAT
            self.answer_proj = nn.Linear(value_dim*self.num_head*3, 1)
        else:
             raise NotImplementedError("Not implemented!")

        # criterion
        # self.criterion = nn.CrossEntropyLoss()
        if 'one-shot' in self.mode:
            self.criterion = LabelSmoothingLoss(2, 0.1)
        elif 'select-one' in self.mode:
            self.criterion = LabelSmoothingLoss(4, 0.1)
        self.softmax = nn.Softmax(dim=-1)

        # Additional Techniques
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, question, answer, text, text_lengths, images, image_lengths, label):
        question = self.que_emb(question) # (bs*4, value_dim*self.num_head)
        answer   = self.ans_emb(answer) # (bs*4, value_dim*self.num_head)

        batch_size = question.shape[0]
        
        if self.mode == 'one-shot':
            outputs = torch.cat([question, answer, question*answer], 1) # (bs*4, value_dim*3)
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            prediction = self.softmax(outputs)[:, 0] # (bs*4)
            loss = self.criterion(outputs, label)
        elif self.mode == 'select-one':
            outputs = torch.cat([question, answer, question*answer], 1) # (bs*4, value_dim*3+key_dim)
            outputs = self.answer_proj(outputs) # (bs*4, 1)
            prediction = self.softmax(outputs.view(-1, 4)) # (bs, 4)
            loss = self.criterion(prediction, torch.LongTensor(np.zeros(prediction.shape[0])).to(self.device))
        else:
            raise NotImplementedError("Not implemented!")

        return prediction, loss