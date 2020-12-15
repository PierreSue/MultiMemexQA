# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU

from __future__ import print_function, division
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random

from model import MemexQA, NaiveQA
from dataset import TrainDataset, DevDataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay, default=5e-6')
parser.add_argument('--manualSeed', type=int, default=1126, help='manual seed')
parser.add_argument('--mode', default='one-shot', help='training experiments')
parser.add_argument('--inpf', default='./new_dataset/', help='folder for input data')
parser.add_argument('--outf', default='./output/', help='folder to output csv and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--keep', action='store_true', help='train the model from the previous spec')
parser.add_argument('--naive', action='store_true', help='train using the naive model')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# Specify cuda
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    device = torch.device("cuda:{}".format(opt.gpu_id))
else:
    device = torch.device("cpu")
    

# collate_fn
def collate_fn(batch):
    # question_embed, answer, choice
    # album_title, album_description, album_when, album_where
    # image_feats, photo_titles, image_lengths
    answer_embed = [item[1].unsqueeze(0) for item in batch] # (bs, 1, 768)
    choice_embed = [item[2] for item in batch] # (bs, 3, 768)
    answer = torch.stack([torch.cat([a, b], 0) for a, b in zip(answer_embed, choice_embed)])

    bs, num_label, text_dim = answer.shape

    question = torch.stack([item[0].unsqueeze(0).repeat(num_label,1) for item in batch]).view(-1, text_dim) # (bs*4, 768) 
    answer   = answer.view(-1, text_dim) # (bs*4, 768)

    album_title  = [item[3] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768) .unsqueeze(0).repeat(num_label,1,1)
    album_desp   = [item[4] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
    album_when   = [item[5] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
    album_where  = [item[6] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
    photo_titles = [item[8] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 768)
    text = [torch.cat([a,b,c,d,e], 0) for a,b,c,d,e in zip(album_title,album_desp,album_when,album_where,photo_titles)] # (bs, ..., 768)
    text_lengths = torch.LongTensor([t.shape[0] for t in text]) # (bs*4)
    text = pad_sequence(text, batch_first=True)

    images = [item[7] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 2537)
    image_lengths = torch.LongTensor([i.shape[0] for i in images]) # (bs*4)
    images = pad_sequence(images, batch_first=True)

    # Label smoothing
    label = torch.LongTensor(bs*([0]+[1]*(num_label-1)))

    return question, answer, text, text_lengths, images, image_lengths, label


# DataSet and DataLoader
train_dataset = TrainDataset(opt.inpf)
dev_dataset   = DevDataset(opt.inpf)

train = DataLoader(train_dataset, batch_size=opt.batchSize, 
                   shuffle=True, num_workers=opt.workers,
                   collate_fn=collate_fn)

dev   = DataLoader(dev_dataset, batch_size=opt.batchSize,
                   shuffle=False, num_workers=opt.workers,
                   collate_fn=collate_fn)

if opt.naive:
    model = NaiveQA(input_dim=768, img_dim=2537, hidden_dim=768, key_dim=64, value_dim=64, num_label=2, num_head=4, num_layer=2, mode=opt.mode, device=device)
else:
    model = MemexQA(input_dim=768, img_dim=2537, hidden_dim=768, key_dim=64, value_dim=64, num_label=2, num_head=4, num_layer=2, mode=opt.mode, device=device)
if opt.keep:
    checkpoint = torch.load('{}/checkpoint_best'.format(opt.outf))
    model.load_state_dict(checkpoint['model_state_dict'])
    last_epoch = checkpoint['epoch']
else:
    last_epoch = 0
model.to(device)
print(model)


# Criterion
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adadelta(model.parameters(), lr=opt.lr, rho=0.9, eps=1e-06, weight_decay=opt.weight_decay)
if opt.keep:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


Acc = 0.0
train_length, best_loss = len(train), float('inf')
for epoch in range(last_epoch, opt.niter):
    # Training
    train_loss, count, acc, acc_count = 0.0, 0, 0., 0
    model.train()
    for id, data in enumerate(train):
        question, answer, text, text_lengths, images, image_lengths, label = data
        
        optimizer.zero_grad()
        
        question = question.to(device)
        answer = answer.to(device)
        images  = images.to(device)
        text = text.to(device)
        label = label.to(device)

        predictions, loss = model(question, answer, text, text_lengths, images, image_lengths, label)
        predictions = torch.argmax(predictions.view(-1, 4), -1).detach().cpu().numpy()

        loss.backward()
        optimizer.step()

        acc += np.sum(predictions == np.zeros_like(predictions))
        acc_count += label.shape[0]//4
    
        train_loss += loss.item()
        count += 1

        if (id+1) % 30 == 0:
            print('Epoch :{}, Progress : {}/{}, Loss:{:.3f}, ACC:{}%, DevLoss:{:.3f}, DevAcc:{}%'.format(epoch+1, id+1, \
                                    train_length, train_loss/count, int(acc/acc_count*100), best_loss, int(Acc*100)))
            train_loss = 0.0
            count      = 0

    model.eval()
    dev_loss, dev_count, acc, acc_count = 0., 0, 0., 0
    with torch.no_grad():
        for id, data in enumerate(dev):
            question, answer, text, text_lengths, images, image_lengths, label = data
            
            question = question.to(device)
            answer = answer.to(device)
            images  = images.to(device)
            text = text.to(device)
            label = label.to(device)
            
            predictions, loss = model(question, answer, text, text_lengths, images, image_lengths, label)
            predictions = torch.argmax(predictions.view(-1, 4), -1).detach().cpu().numpy()
            
            acc += np.sum(predictions == np.zeros_like(predictions))
            acc_count += label.shape[0]//4

            dev_loss += loss.item()
            dev_count += 1
            
        dev_loss = dev_loss/dev_count
        acc = acc/acc_count

        print(acc)
        if acc < Acc:
            lr_scheduler.step()
        else:
            torch.save({ 'epoch': epoch+1, \
                'model_state_dict': model.state_dict(), \
                'optimizer_state_dict': optimizer.state_dict(), \
                'loss': dev_loss, \
                'acc': acc}, \
                '{}/checkpoint_best'.format(opt.outf))
            Acc = acc

        if dev_loss < best_loss:
            best_loss = dev_loss