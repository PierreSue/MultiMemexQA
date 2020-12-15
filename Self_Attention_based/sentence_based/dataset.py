# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU

from __future__ import print_function, division
from collections import defaultdict

import os
import random
import pickle
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, prepropath):
        super(TrainDataset, self).__init__()
        data_path = os.path.join(prepropath,"train_data.p")
        shared_path = os.path.join(prepropath,"train_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.num_examples = len(self.data['questions'])

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']
        
    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], torch.stack(self.data['choices'][idx])
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])
            album_description.append(album['description'])
            album_where.append(album['where'])
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)

        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)

        photo_titles = torch.cat(photo_titles, 0)
        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(), answer.detach(), choice.detach(), \
               album_title.detach(), album_description.detach(), album_when.detach(), album_where.detach(), \
               image_feats.detach(), photo_titles.detach(), image_lengths.detach()

    def __len__(self):
        return self.num_examples


class DevDataset(Dataset):
    def __init__(self, prepropath):
        super(DevDataset, self).__init__()
        data_path = os.path.join(prepropath,"val_data.p")
        shared_path = os.path.join(prepropath,"val_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.num_examples = len(self.data['questions'])

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']

    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], torch.stack(self.data['choices'][idx])
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])
            album_description.append(album['description'])
            album_where.append(album['where'])
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)

        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)

        photo_titles = torch.cat(photo_titles, 0)
        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(), answer.detach(), choice.detach(), \
               album_title.detach(), album_description.detach(), album_when.detach(), album_where.detach(), \
               image_feats.detach(), photo_titles.detach(), image_lengths.detach()

    def __len__(self):
        return self.num_examples


class TestDataset(Dataset):
    def __init__(self, prepropath):
        super(TestDataset, self).__init__()
        data_path = os.path.join(prepropath,"test_data.p")
        shared_path = os.path.join(prepropath,"test_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.num_examples = len(self.data['questions'])

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']

    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], torch.stack(self.data['choices'][idx])
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])
            album_description.append(album['description'])
            album_where.append(album['where'])
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)

        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)

        photo_titles = torch.cat(photo_titles, 0)
        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(), answer.detach(), choice.detach(), \
               album_title.detach(), album_description.detach(), album_when.detach(), album_where.detach(), \
               image_feats.detach(), photo_titles.detach(), image_lengths.detach()
        
    def __len__(self):
        return self.num_examples
    
if __name__ == '__main__':
    train_dataset = TrainDataset('./new_dataset/')
    question_embed, answer, choice, album_title, album_description, album_when, album_where, image_feats, photo_titles, image_lengths = train_dataset[0]
    print(question_embed.shape)
    print(choice.shape)
    print(album_title.shape)
    print(album_description.shape)
    print(album_when.shape)
    print(album_where.shape)
    print(image_feats.shape)
    print(photo_titles.shape)
    print(image_lengths.shape)
    print(image_lengths)
    exit()

    dev_dataset   = DevDataset('./new_dataset/')    
    test_dataset  = TestDataset('./new_dataset/')
    
    print(train_dataset[0])    
    print(dev_dataset[0])    
    print(test_dataset[0])