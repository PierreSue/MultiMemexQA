# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU
# Revised From https://github.com/JunweiLiang/MemexQA_StarterCode

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

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def get_args():
	parser = argparse.ArgumentParser(description="giving the original memoryqa dataset, will generate a *_data.p, *_shared.p for each dataset.")
	parser.add_argument("datajson",type=str,help="path to the qas.json")
	parser.add_argument("albumjson",type=str,help="path to album_info.json")
	parser.add_argument("testids",type=str,help="path to test id list")
	parser.add_argument("--valids",type=str,default=None,help="path to validation id list, if not set will be random 20%% of the training set")

	parser.add_argument("imgfeat",action="store",type=str,help="/path/to img feat npz file")
	parser.add_argument("glove",action="store",type=str,help="/path/to glove vector file")
	parser.add_argument("outpath",type=str,help="output path")

	parser.add_argument("--use_BERT",action="store_true",default=False,help="Pretrained Embedding")

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


# for each token with "-" or others, remove it and split the token 
def process_tokens(tokens):
	newtokens = []
	l = ("-","/", "~", '"', "'", ":","\)","\(","\[","\]","\{","\}")
	for token in tokens:
		# split then add multiple to new tokens
		newtokens.extend([one for one in re.split("[%s]"%("").join(l),token) if one != ""])
	return newtokens


# word_counter words are lowered already
def get_word2vec(args, word_counter, word2vec='glove'):
	word2vec_dict = {}
	if word2vec == 'glove':
		with open(args.glove, 'r', encoding='utf-8') as fh:
			for line in fh:
				array = line.lstrip().rstrip().split(" ")
				word = array[0]
				vector = list(map(float, array[1:]))
				if word in word_counter:
					word2vec_dict[word] = vector
				elif word.lower() in word_counter:
					word2vec_dict[word.lower()] = vector

	elif word2vec == 'bert':
		for word, count in word_counter.items():
			input_ids = tf.constant(tokenizer.encode(word))[None, :]
			outputs = model(input_ids)
			word2vec_dict[word] = tf.reshape(outputs[1], [768]).numpy().tolist()

	return word2vec_dict


def prepro_each(args,data_type,question_ids,start_ratio=0.0,end_ratio=1.0):
	def word_tokenize(tokens):
		# nltk.word_tokenize will split ()
		# "a" -> '``' + a + "''"
		# lizzy's -> lizzy + 's
		# they're -> they + 're
		# then we remove and split "-"
		return process_tokens([token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)])

	qas = {str(qa['question_id']):qa for qa in args.qas}

	global_aids = {} # all the album Id the question used, also how many question used that album

	q, cq, y, cy, aid, qid, cs, ccs, idxs, yidx = [],[],[],[],[],[],[],[],[],[]
	word_counter,char_counter = Counter(),Counter() # lower word counter

	start_idx = int(round(len(question_ids) * start_ratio))
	end_idx = int(round(len(question_ids) * end_ratio))

	for idx, question_id in enumerate(tqdm(question_ids[start_idx:end_idx])):
		qa = qas[question_id]

		# question
		# qa['question'] = 'Where did we travel to ?'
		# qi = ['Where', 'did', 'we', 'travel', 'to', '?']
		# cqi = [['W', 'h', 'e', 'r', 'e'], ['d', 'i', 'd'], ['w', 'e'], ['t', 'r', 'a', 'v', 'e', 'l'], ['t', 'o'], ['?']]
		qi = word_tokenize(qa['question']) # no lower here
		cqi = [list(qij) for qij in qi]

		for qij in qi:
			word_counter[qij.lower()] += 1
			for qijk in qij:
				char_counter[qijk] += 1

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
		yi = word_tokenize(qa['answer'])
		cyi = [list(yij) for yij in yi]
		for yij in yi:
			word_counter[yij.lower()] += 1
			for yijk in yij:
				char_counter[yijk] +=1

		# ci = [['Uis', 'homecoming'], ['Tahoe'], ['Folly', 'beach']]
		# cci = [[['U', 'i', 's'], ['h', 'o', 'm', 'e', 'c', 'o', 'm', 'i', 'n', 'g']], [['T', 'a', 'h', 'o', 'e']], [['F', 'o', 'l', 'l', 'y'], ['b', 'e', 'a', 'c', 'h']]]
		ci = qa['multiple_choices_4'][:] # copy it
		# remove the answer in choices
		yidxi = ci.index(qa['answer']) # this is for during testing, we need to reconstruct the answer in the original order
		ci.remove(qa['answer']) # will error if answer not in choice
		cci = [] # char for choices
		for i,c in enumerate(ci):
			ci[i] = word_tokenize(c)
			cci.append([list(ciij) for ciij in ci[i]])
			for ciij in ci[i]:
				word_counter[ciij.lower()]+=1
				for ciijk in ciij:
					char_counter[ciijk]+=1

		q.append(qi)
		cq.append(cqi)
		y.append(yi)
		cy.append(cyi)
		yidx.append(yidxi)
		cs.append(ci)
		ccs.append(cci)
		aid.append([str(one) for one in qa['album_ids']])
		qid.append(question_id)
		idxs.append(idx) # increament index for each qa

	albums = {str(album['album_id']):album for album in args.albums}
	album_info = {}
	pid2feat = {}
	for albumId in tqdm(global_aids):
		album = albums[albumId]
		used = global_aids[albumId]

		temp = {'aid':album['album_id']}

		# album info
		temp['title'] = word_tokenize(album['album_title'])
		temp['title_c'] = [list(tok) for tok in temp['title']]
		temp['description'] = word_tokenize(strip_tags(album['album_description']))
		temp['description_c'] = [list(tok) for tok in temp['description']]

		# use _ to connect?
		if album['album_where'] is None:
			temp['where'] = []
			temp['where_c'] = []
		else:
			temp['where'] = word_tokenize(album['album_where'])
			temp['where_c'] = [list(tok) for tok in temp['where']]
		temp['when'] = word_tokenize(album['album_when'])
		temp['when_c'] = [list(tok) for tok in temp['when']]

		# photo info
		temp['photo_urls'] = [url for url in album['photo_urls']]
		temp['photo_titles'] = [word_tokenize(title) for title in album['photo_titles']]
		temp['photo_titles_c'] = [[list(tok) for tok in title] for title in temp['photo_titles']]


		temp['photo_ids'] = [str(pid) for pid in album['photo_ids']]
		for pid in temp['photo_ids']:
			if pid not in pid2feat:
				pid2feat[pid] = args.images[pid]

		for t in temp['title'] + temp['description'] + temp['where'] + temp['when'] + [tok for title in temp['photo_titles'] for tok in title ]:
			word_counter[t.lower()] += used
			for c in t:
				char_counter[c] += used
		
		album_info[albumId] = temp

	if args.use_BERT:
		word2vec_dict = get_word2vec(args, word_counter, "bert")
	else:
		word2vec_dict = get_word2vec(args, word_counter, "glove")

	#q,cq,y,cy,aid,qid,cs,ccs,idxs 
	data = {
		'q':q,
		'cq':cq,
		'y':y,
		'cy':cy,
		'yidx': yidx,# the original answer idx in the choices list # this means the correct index
		'aid':aid, # each is a list of aids
		'qid':qid,
		'idxs':idxs,
		'cs':cs, # each is a list of wrong choices
		'ccs':ccs,
	}

	shared = {
		"albums" :album_info, # albumId -> photo_ids/title/when/where ...
		"pid2feat":pid2feat, # pid -> image feature
		"wordCounter":word_counter,
		"charCounter":char_counter,
		"word2vec":word2vec_dict
	}
	print("data:%s, char entry:%s, word entry:%s, word2vec entry:%s,album: %s/%s, image_feat:%s"%(data_type,\
			len(char_counter),len(word_counter),len(word2vec_dict),len(album_info),len(albums),len(pid2feat)))

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

	prepro_each(args,"train",trainIds,0.0,1.0)
	prepro_each(args,"val",valIds,0.0,1.0)
	prepro_each(args,"test",testIds,0.0,1.0)