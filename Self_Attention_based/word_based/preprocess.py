# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU

import re
import os
import sys
import json
import nltk
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

# import cPickle as pickle
import pickle

import argparse
from html.parser import HTMLParser

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def get_args():
	parser = argparse.ArgumentParser(description="giving the original memoryqa dataset, will generate a *_data.p, *_shared.p for each dataset.")
	parser.add_argument("datajson",type=str,help="path to the qas.json")
	parser.add_argument("albumjson",type=str,help="path to album_info.json")
	parser.add_argument("testids",type=str,help="path to test id list")
	parser.add_argument("--valids",type=str,default=None,help="path to validation id list, if not set will be random 20%% of the training set")

	parser.add_argument("imgfeat",action="store",type=str,help="/path/to img feat npz file")
	parser.add_argument("outpath",type=str,help="output path")
	parser.add_argument("--word_based",action="store_true",default=False,help="Word-based Embedding")


	return parser.parse_args()


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def l2norm(feat):
	l2norm = np.linalg.norm(feat,2)
	return feat/l2norm


# HTML 
class MLStripper(HTMLParser):
	def __init__(self):
		super().__init__()
		self.reset()
		self.fed = []
	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)
		
def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data()


# word_counter words are lowered already
def sentence2vec(sentence, word_based=False):
	if word_based:
		try:
			inputs = tokenizer(sentence, return_tensors="pt")
			outputs = model(**inputs)['last_hidden_state']
			return outputs[:,1:-1,:].view(-1, 768)
		except:
			return torch.zeros((1, 768))
	else:
		try:
			inputs = tokenizer(sentence, return_tensors="pt")
			outputs = model(**inputs)['pooler_output']
			return outputs.view(-1)
		except:
			return torch.zeros(768)


def prepro_each(args, data_type, question_ids):
	qas = {str(qa['question_id']):qa for qa in args.qas}

	global_aids = {} # all the album Id the question used, also how many question used that album

	questions, questions_embed, answers = [], [], []
	aid, qid, choices = [], [], []
	yidx = []

	for idx, question_id in enumerate(tqdm(question_ids)):
		qa = qas[question_id]

		# question
		# qa['question'] = 'Where did we travel to ?'
		# qi = ['Where', 'did', 'we', 'travel', 'to', '?']
		# cqi = [['W', 'h', 'e', 'r', 'e'], ['d', 'i', 'd'], ['w', 'e'], ['t', 'r', 'a', 'v', 'e', 'l'], ['t', 'o'], ['?']]
		question = qa['question']
		question_embed = sentence2vec(qa['question'], args.word_based)

		# album ids
		for albumId in qa['album_ids']:
			albumId = str(albumId)
			if albumId not in global_aids:
				global_aids[albumId] = 0
			global_aids[albumId]+=1 # remember how many times this album is used

		# answer, choices
		# qa['answer'] = 'waco'
		# yi = ['Waco']
		# cyi = [['W', 'a', 'c', 'o']]
		answer = sentence2vec(qa['answer'], args.word_based)

		# ci = [['Uis', 'homecoming'], ['Tahoe'], ['Folly', 'beach']]
		# cci = [[['U', 'i', 's'], ['h', 'o', 'm', 'e', 'c', 'o', 'm', 'i', 'n', 'g']], [['T', 'a', 'h', 'o', 'e']], [['F', 'o', 'l', 'l', 'y'], ['b', 'e', 'a', 'c', 'h']]]
		choice = qa['multiple_choices_4'][:] # copy it
		# remove the answer in choices
		yidxi = choice.index(qa['answer']) # this is for during testing, we need to reconstruct the answer in the original order
		choice.remove(qa['answer']) # will error if answer not in choice
		choice = [sentence2vec(c, args.word_based) for c in choice]

		# Append
		questions.append(question)
		questions_embed.append(question_embed)
		answers.append(answer)

		aid.append([str(album_id) for album_id in qa['album_ids']])
		qid.append(question_id)
		choices.append(choice)
		yidx.append(yidxi)

	albums = {str(album['album_id']):album for album in args.albums}
	album_info, pid2feat = {}, {}
	for albumId in tqdm(global_aids):
		album = albums[albumId]
		temp = {'aid':album['album_id']}

		# album info
		temp['title'] = sentence2vec(album['album_title'])
		temp['description'] = sentence2vec(strip_tags(album['album_description']))

		# use _ to connect?
		if album['album_where'] is None:
			temp['where'] = torch.zeros(768)
		else:
			temp['where'] = sentence2vec(album['album_where'])
		temp['when'] = sentence2vec(album['album_when'])

		# photo info
		temp['photo_urls'] = [url for url in album['photo_urls']]
		temp['photo_titles'] = [sentence2vec(title) for title in album['photo_titles']]
		temp['photo_ids'] = [str(pid) for pid in album['photo_ids']]

		for pid in temp['photo_ids']:
			if pid not in pid2feat:
				pid2feat[pid] = args.images[pid]
		
		album_info[albumId] = temp
 
	data = {
		'questions':questions,
		'questions_embed':questions_embed,
		'answers':answers,
		'yidx': yidx,
		'aid':aid,
		'qid':qid,
		'choices':choices
	}

	shared = {
		"albums" :album_info, # albumId -> photo_ids/title/when/where ...
		"pid2feat":pid2feat, # pid -> image feature
	}

	print("data:{},album: {}/{}, image_feat:{}".format(data_type, len(album_info), len(albums), len(pid2feat)))

	with open(os.path.join(args.outpath,"%s_data.p"%data_type), "wb") as fp:
		pickle.dump(data, fp)
	with open(os.path.join(args.outpath,"%s_shared.p"%data_type), "wb") as fp:
		pickle.dump(shared, fp)



def getTrainValIds(qas,validlist,testidlist):
	testIds = [one.strip() for one in open(testidlist,"r").readlines()]

	valIds = []
	if validlist is not None:
		valIds = [one.strip() for one in open(validlist,"r").readlines()]

	trainIds = []
	for one in qas:
		qid = str(one['question_id'])
		if((qid not in testIds) and (qid not in valIds)):
			trainIds.append(qid)

	if validlist is None:
		valcount = int(len(trainIds)*0.2)
		random.seed(1)
		random.shuffle(trainIds)
		random.shuffle(trainIds)
		valIds = trainIds[:valcount]
		trainIds = trainIds[valcount:]
		
	print("total trainId:%s,valId:%s,testId:%s, total qa:%s"%(len(trainIds),len(valIds),len(testIds),len(qas)))
	return trainIds,valIds,testIds



if __name__ == "__main__":
	# albumjson='memexqa_dataset_v1.1/album_info.json'
	# datajson='memexqa_dataset_v1.1/qas.json', 
	# glove='memexqa_dataset_v1.1/glove.6B.100d.txt', 
	# imgfeat='memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz', 
	# outpath='prepro', 
	# testids='memexqa_dataset_v1.1/test_question.ids', 
	# use_BERT=True
	args = get_args()
	mkdir(args.outpath)
	
	# Length = 20563
	# 'evidence_photo_ids': ['1164554'], 
	# 'question': 'What did we do?', 
	# 'secrete_type': 'single_album', 
	# 'multiple_choices_20': ['February 26 2005', 'May 28 2006', 'December 18 2004', 'October 29 2004', 'In bushes', 'Funeral', 'Springer wedding', 'Jacksonville fl', '1 of the kids', 'Georgia may jagger', 'Lauren', 'Rhubarb', 'Watch red sox parade', 'White', "Celebrating liza's birthday", 'Blue stuffed animal', '14', '3', 'Twice', '4 tiers'], 
	# 'multiple_choices_4': ['Watch red sox parade', 'White', "Celebrating liza's birthday", 'Blue stuffed animal'], 
	# 'secrete_hit_id': '3W1K7D6QSB4YZRFASL9VJQP73FUBZL', 
	# 'secrete_aid': '29851', 
	# 'album_ids': ['29851'], 
	# 'answer': 'Watch red sox parade', 
	# 'question_id': 170000, 
	# 'flickr_user_id': '35034354137@N01'
	args.qas = json.load(open(args.datajson,"r"))

	# Length = 630
	# 'album_where': 'New York, 10009, USA', 
	# 'photo_urls': ['https://farm3.staticflickr.com/2762/4513010720_0f5aacacbf_o.jpg', 'https://farm3.staticflickr.com/2127/4513022954_64334f780e_o.jpg', 'https://farm3.staticflickr.com/2024/4513024686_19204de865_o.jpg', 'https://farm3.staticflickr.com/2743/4512372271_bb7b188d47_o.jpg', 'https://farm3.staticflickr.com/2286/4512373755_d71b7d82c9_o.jpg', 'https://farm3.staticflickr.com/2739/4512366991_3ca24f3105_o.jpg'], 
	# 'album_id': '72157623710621031', 
	# 'photo_gps': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 
	# 'photo_titles': ['Blowing Out Candles', 'Eden and Lulu', 'Party Stump', 'Paper Crown I', 'Craft Table', 'Slice of Cake'], 
	# 'album_description': 'April 11, 2010, at La Plaza Cultural community garden.  Lulu was in the 2s class at Little Missionary when Eden was in the 4s, and they just took a shine to each other.', 
	# 'album_when': 'on April 11 2010', 
	# 'flickr_user_id': '10485077@N06', 
	# 'album_title': "Lulu's 4th Birthday", 
	# 'photo_tags': ['birthday nyc newyorkcity eastvillage ny newyork downtown lulu manhattan birthdayparty jc alphabetcity somethingsweet saoirse laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork downtown lulu manhattan birthdayparty eden alphabetcity somethingsweet laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork cake downtown lulu manhattan birthdayparty birthdaycake stump eden alphabetcity somethingsweet laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork downtown princess manhattan birthdayparty crown eden alphabetcity laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork painting downtown lulu manhattan crafts birthdayparty eden alphabetcity crowns laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork cake downtown manhattan birthdayparty birthdaycake eden alphabetcity somethingsweet sliceofcake laplazacultural'], 
	# 'photo_captions': ["at [female] 's birthday party , they ate cake .", 'they had ice cream too .', 'then they got to do some crafts .', '[female] made a silly hat .', 'everybody had fun making things .', 'she made fun art projects .'], 
	# 'photo_ids': ['4513010720', '4513022954', '4513024686', '4512372271', '4512373755', '4512366991']}
	args.albums = json.load(open(args.albumjson,"r"))

	# if the image is a .p file, then we will read it differently
	# Length = 5090
	# map -> id : feature
	# 5739189334 : shape=2537
	if(args.imgfeat.endswith(".p")):
		print("read pickle image feat.")
		imagedata = pickle.load(open(args.imgfeat,"r"))
		args.images = {}
		assert len(imagedata[0]) == len(imagedata[1])
		for i,pid in enumerate(imagedata[0]):
			args.images[pid] = imagedata[1][i]
	else:
		print("read npz image feat.")
		args.images = np.load(args.imgfeat)

	# trainIds = 80% training data
	# valIds   = 20% training data
	# testIds  = args.testids = memexqa_dataset_v1.1/test_question.ids
	trainIds,valIds,testIds = getTrainValIds(args.qas,args.valids,args.testids)

	prepro_each(args,"train",trainIds)
	prepro_each(args,"val",valIds)
	prepro_each(args,"test",testIds)

	
	