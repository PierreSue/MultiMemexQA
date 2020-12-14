# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU
# Revised From https://github.com/JunweiLiang/MemexQA_StarterCode

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from utils import flatten,reconstruct,Dataset,exp_mask
import numpy as np
import random,sys
from transformer import Encoder

VERY_NEGATIVE_NUMBER = -1e30

def get_model(config):
    # implement a multi gpu model?
    with tf.name_scope(config.modelname), tf.device("/gpu:0"):
        model = Model(config,"model_%s"%config.modelname)

    return model


from copy import deepcopy # for C[i].insert(Y[i])



# x -> [Num,JX,W,embedding dim] # conv2d requires an input of 4d [batch, in_height, in_width, in_channels]
def conv1d(x,filter_size,height,keep_prob,is_train=None,wd=None,scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = x.get_shape()[-1] # embedding dim[8]
        filter_var = tf.get_variable("filter",shape=[1,height,num_channels,filter_size],dtype="float")
        bias = tf.get_variable('bias',shape=[filter_size],dtype='float')
        strides = [1,1,1,1]
        # add dropout to input
        
        d = tf.nn.dropout(x,keep_prob=keep_prob)
        outd = tf.cond(is_train,lambda:d,lambda:x)
        #conv
        xc = tf.nn.relu(tf.nn.conv2d(outd,filter_var,strides,padding='VALID')+bias)
        # simple max pooling?
        out = tf.reduce_max(xc,2) # [-1,JX,num_channel]

        if wd is not None:
            add_wd(wd)

        return out

# fully-connected layer
# [N,M,JX,JQ,2d] => x[N*M*JX*JQ,2d] * W[2d,output_size] -> 
def linear(x,output_size,scope,add_tanh=True,wd=None,have_bias=True):
    with tf.variable_scope(scope):
        #x = tf.Print(x, [tf.shape(x)], message="x debug info", summarize=100)
        
        # since the input here is not two rank, we flat the input while keeping the last dims
        keep = 1
        flat_x = flatten(x,keep) # keeping the last one dim # [N,M,JX,JQ,2d] => [N*M*JX*JQ,2d]
        #flat_x = tf.Print(flat_x, [tf.shape(flat_x)], message="flat_x debug info", summarize=100)
        if not (type(output_size) == type(1)): # need to be get_shape()[k].value
            output_size = output_size.value

        W = tf.get_variable("W",dtype="float",initializer=tf.truncated_normal([flat_x.get_shape()[-1],output_size],stddev=0.1))
        #W = tf.Print(W, [tf.shape(W)], message="W debug info", summarize=100)
        if have_bias == True:
            bias_start = 0.0
            bias = tf.get_variable("b",dtype="float",initializer=tf.constant(bias_start,shape=[output_size]))
            flat_out = tf.matmul(flat_x,W)+bias
        else:
            flat_out = tf.matmul(flat_x,W)
        #flat_out = tf.Print(flat_out, [tf.shape(flat_out)], message="flat_out debug info", summarize=100)

        if add_tanh:
            flat_out = tf.tanh(flat_out,name="tanh")


        if wd is not None:
            add_wd(wd)

        out = reconstruct(flat_out,x,keep)
        #out = tf.Print(out, [tf.shape(out)], message="out debug info", summarize=100)

        return out

def similarity(v1,v2):
    return tf.concat([v1*v2,(v1-v2)*(v1-v2)],axis=-1)

def attention2(hinfo, hq, C, scope):
    #hinfo: [N,K,T,2d]
    #hq: [N,JQ,2d]  
    #C: [N,T,T] T=M*JMAX
    with tf.variable_scope(scope):
        #hinfo = tf.Print(hinfo, [tf.shape(hinfo)], message="hinfo debug info", summarize=100)
        K = tf.shape(hinfo)[1]
        T = tf.shape(hinfo)[2]
        JQ = tf.shape(hq)[1]

        hinfo_aug_cross = tf.tile(tf.expand_dims(hinfo,3),[1,1,1,JQ,1]) # [N,K,T,2D] -> [N,K,T,JQ,2D]
        #hinfo_aug_cross = tf.Print(hinfo_aug_cross, [tf.shape(hinfo_aug_cross)], message="hinfo_aug_cross debug info", summarize=100)
        hq_aug = tf.tile(tf.expand_dims(tf.expand_dims(hq,1),1), [1,K,T,1,1]) #[N,JQ,2D] -> [N,K,T,JQ,2d]
        #hq_aug = tf.Print(hq_aug, [tf.shape(hq_aug)], message="hq_aug debug info", summarize=100)
        simi_cross = similarity(hinfo_aug_cross,hq_aug) #[N,K,T,JQ,4D]
        #simi_cross = tf.Print(simi_cross, [tf.shape(simi_cross)], message="simi_cross debug info", summarize=100)

        S = linear(simi_cross,1,scope="s_linear",add_tanh=True,have_bias=True) #[N,K,T,JQ,1]
        S = tf.squeeze(S,-1) #[N,K,T,JQ]
        #S = tf.Print(S, [tf.shape(S)], message="S debug info", summarize=100)
        

        A = tf.reduce_max(S,axis=-1) #[N,K,T,JQ] -> [N,K,T]
        A = tf.nn.softmax(A,axis=-1) #[N,K,T]
        #A = tf.Print(A, [tf.shape(A)], message="A debug info", summarize=100)
        
        A_mul_F = tf.expand_dims(A, -1) * hinfo #[N,K,T,2D]
        A_mul_F = tf.reduce_sum(A_mul_F,-2) #[N,K,2D]
        #A_mul_F = tf.Print(A_mul_F, [tf.shape(A_mul_F)], message="A_mul_F debug info", summarize=100)

        B = tf.reduce_max(tf.reduce_max(S,axis=3),axis=2) #[N,K]
        B = tf.nn.softmax(B,axis=-1) #[N,K]
        #B = tf.Print(B, [tf.shape(B)], message="B debug info", summarize=100)

        h_hat = tf.expand_dims(B,-1) * A_mul_F #[N,K,2D]
        h_hat = tf.reduce_sum(h_hat,-2) #[N,2d]
        #h_hat = tf.Print(h_hat, [tf.shape(h_hat)], message="h_hat debug info", summarize=100)


        D = tf.reduce_max(S,axis=[1,2]) #[N,K,T,JQ] -> [N,JQ]
        D = tf.nn.softmax(D,axis=-1) #[N,JQ]
        #D = tf.Print(D, [tf.shape(D)], message="D debug info", summarize=100)

        q_hat = tf.expand_dims(D,-1) * hq #[N,JQ,2D]
        q_hat = tf.reduce_sum(q_hat,-2) #[N,2d]
        #q_hat = tf.Print(q_hat, [tf.shape(q_hat)], message="q_hat debug info", summarize=100)


        return h_hat,q_hat

def attention1(hinfo, hq, scope, dim_last):
    #hinfo: [N,M,JXA,2d]
    #hq: [N,JQ,2d]
    with tf.variable_scope(scope):
        #window_size = 3
        #hinfo = tf.Print(hinfo, [tf.shape(hinfo)], message="hinfo debug info", summarize=100)
        N = tf.shape(hinfo)[0]
        JQ = tf.shape(hq)[1]
        lq = hq[:,-1,:]
        hinfo = tf.reshape(hinfo,[N,-1,dim_last])  #[N,M,JXA,2d] -> [N,M*JXA,2d]
        T = tf.shape(hinfo)[-2]
        #hinfo = tf.Print(hinfo, [tf.shape(hinfo)], message="hinfo debug info", summarize=100)

        hat_aug = tf.tile(tf.expand_dims(hinfo,2),[1,1,T,1]) #[N,T,T,2d]
        #hat_aug = tf.Print(hat_aug, [tf.shape(hat_aug)], message="hat_aug debug info", summarize=100)
        hat_aug_t = tf.transpose(hat_aug,[0,2,1,3]) #[N,T,T,2d]
        #hat_aug_t = tf.Print(hat_aug_t, [tf.shape(hat_aug_t)], message="hat_aug_t debug info", summarize=100)
        simi = similarity(hat_aug,hat_aug_t) #[N,T,T,4d]
        #simi = tf.Print(simi, [tf.shape(simi)], message="simi debug info", summarize=100)

        #lq_aug = tf.tile(tf.expand_dims(lq,1),[1,T,1]) #[N,2d] -> [N,T,2d]
        #lq_aug = tf.tile(tf.expand_dims(lq_aug,2),[1,1,T,1]) #[N,T,2d] -> [N,T,T,2d]
        lq_aug = tf.tile(tf.expand_dims(tf.expand_dims(lq, 1), 1), [1, T, T, 1]) #[N,2d] -> [N,T,T,2d]
        #lq_aug = tf.Print(lq_aug, [tf.shape(lq_aug)], message="lq_aug debug info", summarize=100)
        C = linear(simi,dim_last,scope="c_linear1",add_tanh=False,have_bias=False) + lq_aug #[N,T,T,2d]
        #C = tf.Print(C, [tf.shape(C)], message="C debug info", summarize=100)
        C = linear(C,1,scope="c_linear2",add_tanh=False,have_bias=False) #[N,T,T,1]
        #C = tf.Print(C, [tf.shape(C)], message="C debug info", summarize=100)
        C = tf.squeeze(C,-1) #[N,T,T]
        #C = tf.Print(C, [tf.shape(C)], message="C debug info", summarize=100)
        C = tf.tanh(C) #[N,T,T]

        F = tf.matmul(C, hinfo) #[N,T,T] mul [N,T,2d] -> [N,T,2d]
        #F = tf.Print(F, [tf.shape(F)], message="F debug info", summarize=100) #[N,T,2d]

        F_aug_cross = tf.tile(tf.expand_dims(F,1),[1,JQ,1,1]) #[N,JQ,T,2d]
        #F_aug_cross = tf.Print(F_aug_cross, [tf.shape(F_aug_cross)], message="F_aug_cross debug info", summarize=100)
        hq_aug = tf.tile(tf.expand_dims(hq,2),[1,1,T,1]) #[N,JQ,T,2d]
        #hq_aug = tf.Print(hq_aug, [tf.shape(hq_aug)], message="hq_aug debug info", summarize=100)
        simi_cross = similarity(F_aug_cross,hq_aug)
        #simi_cross = tf.Print(simi_cross, [tf.shape(simi_cross)], message="simi_cross debug info", summarize=100)


        S = linear(simi_cross,1,scope="s_linear",add_tanh=True,have_bias=True) #[N,JQ,T,1]
        #S = tf.Print(S, [tf.shape(S)], message="S debug info", summarize=100)
        S = tf.squeeze(S,-1) #[N,JQ,T]
        #S = tf.Print(S, [tf.shape(S)], message="S debug info", summarize=100)
        
        A = tf.reduce_max(S,axis=1) # [N,JQ,T] -> [N,T]
        #A = tf.Print(A, [tf.shape(A)], message="A debug info", summarize=100)
        A = tf.nn.softmax(A,axis=-1) #[N,T]
        #A = tf.Print(A, [tf.shape(A)], message="A debug info", summarize=100)
        
        A_mul_F = tf.matmul(tf.transpose(F,[0,2,1]),tf.expand_dims(A,2)) #[N,T,2d]' mul [N,T,1] -> [N,2d,1]
        A_mul_F = tf.squeeze(A_mul_F,-1) #[N,2d]
        #A_mul_F = tf.Print(A_mul_F, [tf.shape(A_mul_F)], message="A_mul_F debug info", summarize=100)

        B = tf.reduce_max(tf.reduce_max(S,axis=1),axis=1) #[N]
        #B = tf.Print(B, [tf.shape(B)], message="B debug info", summarize=100)

        Q = hq #[N,JQ,2d]
        #Q = tf.Print(Q, [tf.shape(Q)], message="Q debug info", summarize=100)
        A_q = tf.reduce_max(S,axis=2) #[N,JQ]
        #A_q = tf.Print(A_q, [tf.shape(A_q)], message="A_q debug info", summarize=100)
        A_q = tf.nn.softmax(A_q,axis=-1) #[N,JQ]
        #A_q = tf.Print(A_q, [tf.shape(A_q)], message="A_q debug info", summarize=100)

        A_mul_Q_q = tf.matmul(tf.transpose(Q, [0,2,1]),tf.expand_dims(A_q,2)) # [N,JQ,2d]' mul [N,JQ,1] -> [[N,2d,1]
        A_mul_Q_q = tf.squeeze(A_mul_Q_q,-1) #[N,2d]
        #A_mul_Q_q = tf.Print(A_mul_Q_q, [tf.shape(A_mul_Q_q)], message="A_mul_Q_q debug info", summarize=100)

        B_q = tf.reduce_max(tf.reduce_max(S,axis=2),axis=1) #[N]
        #B_q = tf.Print(B_q, [tf.shape(B_q)], message="B_q debug info", summarize=100)

        return A_mul_F,B,A_mul_Q_q,B_q


# add current scope's variable's l2 loss to loss collection
def add_wd(wd,scope=None):
    if wd != 0.0:
        scope = scope or tf.get_variable_scope().name
        vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        with tf.variable_scope("weight_decay"):
            for var in vars_:
                weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="%s/wd"%(var.op.name))
                tf.add_to_collection("losses",weight_decay)


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer

class Model():
    def __init__(self,config,scope):
        self.scope = scope
        self.config = config
        # a step var to keep track of current training process
        self.global_step = tf.get_variable('global_step',shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False) # a counter

        # get all the dimension here
        N = self.N = config.batch_size
        
        VW = self.VW = config.word_vocab_size
        VC = self.VC = config.char_vocab_size
        W = self.W = config.max_word_size

        # embedding dim
        self.cd,self.wd,self.cwd = config.char_emb_size,config.word_emb_size,config.char_out_size

        # image dimension
        self.idim = config.image_feat_dim

        self.num_choice = 4

        # step limits
        # M -> album max num
        # -----JX -> title max words (album title,photo title)
        # JXA -> album title max words
        # JXP -> photo title max words
        # JD -> album description max word
        # JT -> album when max word
        # JG -> album where max word

        # JI -> album max photo

        # JA -> max answer (choice) length
        # JQ -> max question length

        # all the inputs

        # album title
        # [N,M,JXA]
        self.at = tf.placeholder('int32',[N,None,None],name="at")
        self.at_c = tf.placeholder("int32",[N,None,None,W],name="at_c")
        self.at_mask = tf.placeholder("bool",[N,None,None],name="at_mask") # to get the sequence length

        # album description
        # [N,M,JD]
        self.ad = tf.placeholder('int32',[N,None,None],name="ad")
        self.ad_c = tf.placeholder("int32",[N,None,None,W],name="ad_c")
        self.ad_mask = tf.placeholder("bool",[N,None,None],name="ad_mask")

        # album when, where
        # [N,M,JT/JG]
        self.when = tf.placeholder("int32",[N,None,None],name="when")
        self.when_c = tf.placeholder("int32",[N,None,None,W],name="when_c")
        self.when_mask = tf.placeholder("bool",[N,None,None],name="when_mask")
        self.where = tf.placeholder("int32",[N,None,None],name="where")
        self.where_c = tf.placeholder("int32",[N,None,None,W],name="where_c")
        self.where_mask = tf.placeholder("bool",[N,None,None],name="where_mask")

        # photo titles
        # [N,M,JI,JXP]
        self.pts = tf.placeholder('int32',[N,None,None,None],name="pts")
        self.pts_c = tf.placeholder("int32",[N,None,None,None,W],name="pts_c")
        self.pts_mask = tf.placeholder("bool",[N,None,None,None],name="pts_mask")

        # for vis
        self.JXP = tf.shape(self.pts)[3]


        # photo
        # [N,M,JI] # each is a photo index
        self.pis = tf.placeholder('int32',[N,None,None],name="pis")
        self.pis_mask = tf.placeholder("bool",[N,None,None],name="pis_mask")

        # question
        self.q = tf.placeholder('int32',[N,None],name="q")
        self.q_c = tf.placeholder('int32', [N, None, W], name='q_c')
        self.q_mask = tf.placeholder("bool",[N,None],name="q_mask")

        # answer + choice words
        # [N,4,JA]
        self.choices = tf.placeholder("int32",[N,self.num_choice,None],name="choices")
        self.choices_c = tf.placeholder("int32",[N,self.num_choice,None,W],name="choices_c")
        self.choices_mask = tf.placeholder("bool",[N,self.num_choice,None],name="choices_mask")

        # 4 choice classification
        self.y = tf.placeholder('bool', [N, self.num_choice], name='y')
        

        # feed in the pretrain word vectors for all batch
        self.existing_emb_mat = tf.placeholder('float',[None,config.word_emb_size],name="pre_emb_mat")

        # feed in the image feature for this batch
        # [photoNumForThisBatch,image_dim]
        # TODO: put image_mat in the gloabl shared big matrix
        self.image_emb_mat = tf.placeholder("float",[None,config.image_feat_dim],name="image_emb_mat")

        # used for drop out switch
        self.is_train = tf.placeholder('bool', [], name='is_train')

        # forward output
        # the following will be added in build_forward and build_loss()
        self.logits = None

        self.yp = None # prob

        self.loss = None

        self.build_forward()
        self.build_loss()

        self.var_ema = None


    def build_forward(self):
        config = self.config
        VW = self.VW
        VC = self.VC
        W = self.W
        N = self.N
        # dynamic decide some step, for sequence length
        M = tf.shape(self.pis)[1] # photo num
        JXA = tf.shape(self.at)[2] # for album title, photo title
        JD = tf.shape(self.ad)[2] # description length
        JT = tf.shape(self.when)[2]
        JG = tf.shape(self.where)[2]

        JI = tf.shape(self.pis)[2] # used for photo_title, photo
        JXP = tf.shape(self.pts)[3]


        JQ = tf.shape(self.q)[1]
        JA = tf.shape(self.choices)[2]

        # embeding size
        cdim,wdim,cwdim = self.cd,self.wd,self.cwd #cwd: char -> word output dimension
        # image feature dim
        idim = self.idim # image_feat dimension

        d = config.hidden_size

        # all input:
        #	at, ad, when, where, 
        #	pts, pis
        #	q, choices

        # embedding
        with tf.variable_scope('emb'):
            # char stuff
            if config.use_char:
                with tf.variable_scope("var"): 
                    char_emb = tf.get_variable("char_emb",shape=[VC,cdim],dtype="float")

                # the embedding for each of character 
                # [N,M,JXA,W]
                Aat_c = tf.nn.embedding_lookup(char_emb,self.at_c) #[N,M,JXA,W] -> [N,M,JXA,W,char_embd_size] 
                #Aat_c = tf.Print(Aat_c, [tf.shape(Aat_c)], message="Aat_c debug info", summarize=100)
                
                # [N,M,JD,W]
                Aad_c = tf.nn.embedding_lookup(char_emb,self.ad_c)
                # [N,M,JT,W]
                Awhen_c = tf.nn.embedding_lookup(char_emb,self.when_c)
                # [N,M,JG,W]
                Awhere_c = tf.nn.embedding_lookup(char_emb,self.where_c)
                # [N,M,JI,JXP,W] -> [N,M,JI,JXP,W,cdim]
                Apts_c = tf.nn.embedding_lookup(char_emb,self.pts_c)

                # [N,JQ,W]
                Aq_c = tf.nn.embedding_lookup(char_emb,self.q_c)
                Achoices_c = tf.nn.embedding_lookup(char_emb,self.choices_c)

                # flatten for conv2d input like images
                Aat_c = tf.reshape(Aat_c,[-1,JXA,W,cdim]) #[N,M,JXA,W,char_embd_size] -> [N*M, JXA, W, char_embd_size]
                #Aat_c = tf.Print(Aat_c, [tf.shape(Aat_c)], message="Aat_c_flat debug info", summarize=100)
                
                Aad_c = tf.reshape(Aad_c,[-1,JD,W,cdim])
                Awhen_c = tf.reshape(Awhen_c,[-1,JT,W,cdim])
                Awhere_c = tf.reshape(Awhere_c,[-1,JG,W,cdim])
                # [N*M*JI,JXP,W,cdim]
                Apts_c = tf.reshape(Apts_c,[-1,JXP,W,cdim])

                Aq_c = tf.reshape(Aq_c,[-1,JQ,W,cdim])
                # [N*4,]
                Achoices_c = tf.reshape(Achoices_c,[-1,JA,W,cdim])

                #char CNN
                filter_size = cwdim # output size for each word
                filter_height = 5
                with tf.variable_scope("conv"):
                    #[N*M, JXA, W, char_embd_size] -> [N*M, JXA, char_emb_size]
                    xat = conv1d(Aat_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d") 
                    #xat = tf.Print(xat, [tf.shape(xat)], message="xat_conv debug info", summarize=100)
                    
                    tf.get_variable_scope().reuse_variables()
                    xad = conv1d(Aad_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
                    xwhen = conv1d(Awhen_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
                    xwhere = conv1d(Awhere_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
                    xpts = conv1d(Apts_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
                    qq = conv1d(Aq_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
                    qchoices = conv1d(Achoices_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")

                    # reshape them back
                    xat = tf.reshape(xat,[-1,M,JXA,cwdim]) #[N*M, JXA, char_emb_size] -> [N,M, JXA, char_emb_size]
                    #xat = tf.Print(xat, [tf.shape(xat)], message="xat_reshape debug info", summarize=100)

                    xad = tf.reshape(xad,[-1,M,JD,cwdim])
                    xwhen = tf.reshape(xwhen,[-1,M,JT,cwdim])
                    xwhere = tf.reshape(xwhere,[-1,M,JG,cwdim])
                    xpts = tf.reshape(xpts,[-1,M,JI,JXP,cwdim])

                    qq = tf.reshape(qq,[-1,JQ,cwdim])
                    # [N,num_choice,JA,cwdim]
                    qchoices = tf.reshape(qchoices,[-1,self.num_choice,JA,cwdim])

            # word stuff
            with tf.variable_scope('word'):
                with tf.variable_scope("var"):
                    # get the word embedding for new words
                    if config.is_train:
                        # for new word
                        word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW,wdim],initializer=get_initializer(config.emb_mat)) # it's just random initialized
                    else: # save time for loading the emb during test
                        word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW,wdim])
                    # concat with pretrain vector
                    # so 0 - VW-1 index for new words, the rest for pretrain vector
                    # and the pretrain vector is fixed
                    word_emb_mat = tf.concat([word_emb_mat,self.existing_emb_mat],0)

                #[N,M,JXA] -> [N,M,JXA,wdim]
                Aat = tf.nn.embedding_lookup(word_emb_mat,self.at)
                Aad = tf.nn.embedding_lookup(word_emb_mat,self.ad)
                Awhen = tf.nn.embedding_lookup(word_emb_mat,self.when)
                Awhere = tf.nn.embedding_lookup(word_emb_mat,self.where)
                Apts = tf.nn.embedding_lookup(word_emb_mat,self.pts)

                Aq = tf.nn.embedding_lookup(word_emb_mat,self.q)
                Achoices = tf.nn.embedding_lookup(word_emb_mat,self.choices)

            # concat char and word
            if config.use_char:
                #xat:[N,M, JXA, char_emb_size]
                #xat = tf.Print(xat, [tf.shape(xat)], message="xat debug info", summarize=100)
                #Aat:[N,M,JXA,wdim]
                #Aat = tf.Print(Aat, [tf.shape(Aat)], message="Aat debug info", summarize=100)
                
                #xat:[N,M,JXA,cwdim+wdim]
                xat = tf.concat([xat,Aat],3)
                #xat = tf.Print(xat, [tf.shape(xat)], message="xat_cat debug info", summarize=100)
                xad = tf.concat([xad,Aad],3)
                xwhen = tf.concat([xwhen,Awhen],3)
                xwhere = tf.concat([xwhere,Awhere],3)
                # [N,M,JI,JX,wdim+cwdim]
                xpts = tf.concat([xpts,Apts],4)

                # [N,JQ,wdim+cwdim]
                qq = tf.concat([qq,Aq],2)
                qchoices = tf.concat([qchoices,Achoices],3)

            else:
                xat = Aat
                xad = Aad
                xwhen = Awhen
                xwhere = Awhere
                xpts = Apts

                qq = Aq
                qchoices = Achoices
                # all the above last dim is the same [wdim+cwdim] or just [wdim]

            # get the image feature
            with tf.variable_scope("image"):

                # [N,M,JI] -> [N,M,JI,idim]
                xpis = tf.nn.embedding_lookup(self.image_emb_mat,self.pis)

                # [N,M,JI,idim] -> [N,M,JI,2d]
                if config.image_linear:
                    xpis = linear(xpis, 2*d, scope = "image_linear",add_tanh=True,have_bias=True)
                
                #xpis = tf.Print(xpis, [tf.shape(xpis)], message="xpis debug info", summarize=100)

        cell_text = tf.nn.rnn_cell.BasicLSTMCell(d,state_is_tuple=True)
        cell_img = tf.nn.rnn_cell.BasicLSTMCell(d,state_is_tuple=True)
        # add dropout
        keep_prob = tf.cond(self.is_train,lambda:tf.constant(config.keep_prob),lambda:tf.constant(1.0))

        cell_text = tf.nn.rnn_cell.DropoutWrapper(cell_text,keep_prob)

        cell_img = tf.nn.rnn_cell.DropoutWrapper(cell_img,keep_prob)

        
        # it is important to think about which LSTM shared with which?

        # sequence length for each
        at_len = tf.reduce_sum(tf.cast(self.at_mask,"int32"),2) # [N,M] # each album's title length
        ad_len = tf.reduce_sum(tf.cast(self.ad_mask,"int32"),2)
        when_len = tf.reduce_sum(tf.cast(self.when_mask,"int32"),2)
        where_len = tf.reduce_sum(tf.cast(self.where_mask,"int32"),2) # [N,M]

        pis_len = tf.reduce_sum(tf.cast(self.pis_mask,"int32"),2) #[N,M,JI] #[N,M]

        pts_len = tf.reduce_sum(tf.cast(self.pts_mask,"int32"),3) # [N,M,JI,JXP] -> [N,M,JI]

        q_len = tf.reduce_sum(tf.cast(self.q_mask,"int32"),1) # [N] # each question 's length

        choices_len = tf.reduce_sum(tf.cast(self.choices_mask,"int32"),2) # [N,4]

        # xat -> [N,M,JXA,wdim+cwdim]
        # xad -> [N,M,JD,wdim+cwdim]
        # xwhen/xwhere -> [N,M,JT/JG,wdim+cwdim]
        # xpts -> [N,M,JI,JXP,wdim+cwdim]

        # xpis -> [N,M,JI,idim]

        # qq -> [N,JQ,wdim+cwdim]
        # qchoices -> [N,4,JA,wdim+cwdim]

        # roll the sentence into lstm for context and question
        # from [N,M,JI,JX] -> [N,M,2d]
        with tf.variable_scope("reader"):
            with tf.variable_scope("text"):
                (fw_hq,bw_hq),(fw_lq,bw_lq) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,qq,sequence_length=q_len,dtype="float",scope="utext")
                # concat the fw and backward lstm output
                hq = tf.concat([fw_hq,bw_hq],2)
                lq = tf.concat([fw_lq.h,bw_lq.h],1) #LSTM CELL

                tf.get_variable_scope().reuse_variables()

                # flat all
                # choices
                flat_qchoices = flatten(qchoices,2) # [N,4,JA,dim] -> [N*4,JA,dim]
                # album title
                #xat = tf.Print(xat, [tf.shape(xat)], message="xat debug info", summarize=100)
                flat_xat = flatten(xat,2) #[N,M,JXA,dim] -> [N*M,JXA,dim]
                flat_xad = flatten(xad,2)
                flat_xwhen = flatten(xwhen,2)
                flat_xwhere = flatten(xwhere,2)
                
                #print "flat_xpis shape:%s"%(flat_xpis.get_shape())

                # photo tiles
                flat_xpts = flatten(xpts,2) # [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
                #print "flat_xpts shape:%s"%(flat_xpts.get_shape())

                # get the sequence length, all one dim
                flat_qchoices_len = flatten(choices_len,0) # [N*4]
                flat_xat_len = flatten(at_len,0) # [N*M]
                flat_xad_len = flatten(ad_len,0) # [N*M]
                flat_xwhen_len = flatten(when_len,0) # [N*M]
                flat_xwhere_len = flatten(where_len,0) # [N*M]
                
                flat_xpts_len = flatten(pts_len,0) # [N*M*JI]
                
                # put all through LSTM
                # uncomment to use ALL LSTM output or LAST LSTM output

                # album title
                # [N*M,JXA,d]
                #flat_xat = tf.Print(flat_xat, [tf.shape(flat_xat)], message="flat_xat debug info", summarize=100)
                #flat_xat_len = tf.Print(flat_xat_len, [tf.shape(flat_xat_len)], message="flat_xat_len debug info", summarize=100)
                (fw_hat_flat,bw_hat_flat),(fw_lat_flat,bw_lat_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xat,sequence_length=flat_xat_len,dtype="float",scope="utext")
                
               
                
                fw_hat = reconstruct(fw_hat_flat,xat,2) # 
                bw_hat = reconstruct(bw_hat_flat,xat,2)
                hat = tf.concat([fw_hat,bw_hat],3) # [N,M,JXA,2d]
                fw_lat = tf.reshape(fw_lat_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
                bw_lat = tf.reshape(bw_lat_flat.h,[N,M,d])
                lat = tf.concat([fw_lat,bw_lat],2) # [N,M,2d]


                # album desciption
                # [N*M,JD,d]
                (fw_had_flat,bw_had_flat),(fw_lad_flat,bw_lad_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xad,sequence_length=flat_xad_len,dtype="float",scope="utext")
                fw_had = reconstruct(fw_had_flat,xad,2) # 
                bw_had = reconstruct(bw_had_flat,xad,2)
                had = tf.concat([fw_had,bw_had],3) # [N,M,JD,2d]
                fw_lad = tf.reshape(fw_lad_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
                bw_lad = tf.reshape(bw_lad_flat.h,[N,M,d])
                lad = tf.concat([fw_lad,bw_lad],2) # [N,M,2d]

                # when
                (fw_hwhen_flat,bw_hwhen_flat),(fw_lwhen_flat,bw_lwhen_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xwhen,sequence_length=flat_xwhen_len,dtype="float",scope="utext")
                fw_hwhen = reconstruct(fw_hwhen_flat,xwhen,2) # 
                bw_hwhen = reconstruct(bw_hwhen_flat,xwhen,2)
                hwhen = tf.concat([fw_hwhen,bw_hwhen],3) # [N,M,JT,2d]
                # LSTM
                fw_lwhen = tf.reshape(fw_lwhen_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
                bw_lwhen = tf.reshape(bw_lwhen_flat.h,[N,M,d])
                lwhen = tf.concat([fw_lwhen,bw_lwhen],2) # [N,M,2d]

                # where
                (fw_hwhere_flat,bw_hwhere_flat),(fw_lwhere_flat,bw_lwhere_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xwhere,sequence_length=flat_xwhere_len,dtype="float",scope="utext")
                fw_hwhere = reconstruct(fw_hwhere_flat,xwhere,2) # 
                bw_hwhere = reconstruct(bw_hwhere_flat,xwhere,2)
                hwhere = tf.concat([fw_hwhere,bw_hwhere],3) # [N,M,JG,2d]
                fw_lwhere = tf.reshape(fw_lwhere_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
                bw_lwhere = tf.reshape(bw_lwhere_flat.h,[N,M,d])
                lwhere = tf.concat([fw_lwhere,bw_lwhere],2) # [N,M,2d]


                # photo title
                # [N*M*JI,JXP,d]
                (fw_hpts_flat,bw_hpts_flat),(fw_lpts_flat,bw_lpts_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xpts,sequence_length=flat_xpts_len,dtype="float",scope="utext")
                fw_hpts = reconstruct(fw_hpts_flat,xpts,2) # 
                bw_hpts = reconstruct(bw_hpts_flat,xpts,2) # [N,M,JI,JXP,d]
                hpts = tf.concat([fw_hpts,bw_hpts],4) # [N,M,JI,JXP,2d]
                # LSTM
                fw_lpts = tf.reshape(fw_lpts_flat.h,[N,M,JI,d]) # [N*M*JI,d] -> [N,M,JI,d]
                bw_lpts = tf.reshape(bw_lpts_flat.h,[N,M,JI,d])
                lpts = tf.concat([fw_lpts,bw_lpts],3) # [N,M,JI,2d]	

                # choices
                (fw_hchoices_flat,bw_hchoices_flat),(fw_lchoices_flat,bw_lchoices_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_qchoices,sequence_length=flat_qchoices_len,dtype="float",scope="utext")
                fw_hchoices = reconstruct(fw_hchoices_flat,qchoices,2) # 
                bw_hchoices = reconstruct(bw_hchoices_flat,qchoices,2)
                hchoices = tf.concat([fw_hchoices,bw_hchoices],3) # [N,4,JA,2d]
                # LSTM
                fw_lchoices = tf.reshape(fw_lchoices_flat.h,[N,-1,d]) # [N*4,d] -> [N,4,d]
                bw_lchoices = tf.reshape(bw_lchoices_flat.h,[N,-1,d])
                lchoices = tf.concat([fw_lchoices,bw_lchoices],2) # [N,4,2d]


            with tf.variable_scope("image"):
                # photos
                flat_xpis = flatten(xpis,2) # [N,M,JI,idim] -> [N*M,JI,idim]
                flat_xpis_len = flatten(pis_len,0) # [N*M]

                if config.use_transformer:
                    #transformer encoder
                    encoder_image = Encoder(num_layers=6, d_model=200, num_heads=8, dff=2048,)
                    image_result = encoder_image(flat_xpis,training=True,mask=None)
                    
                    hpis = reconstruct(image_result,xpis,2) #[N,M,JI,2d]
                    lpis = hpis[:,:,-1,:]
                    #hpis = tf.Print(hpis, [tf.shape(hpis)], message="hpis debug info", summarize=100)
                    #lpis = tf.Print(lpis, [tf.shape(lpis)], message="lpis debug info", summarize=100) #[N,M,2d]
                else:
                    #LSTM
                    (fw_hpis_flat,bw_hpis_flat),(fw_lpis_flat,bw_lpis_flat) = tf.nn.bidirectional_dynamic_rnn(cell_img,cell_img,flat_xpis,sequence_length=flat_xpis_len,dtype="float",scope="uimage")
                    fw_hpis = reconstruct(fw_hpis_flat,xpis,2) # 
                    bw_hpis = reconstruct(bw_hpis_flat,xpis,2) # [N,M,JI,JXP,d]
                    hpis = tf.concat([fw_hpis,bw_hpis],3) # [N,M,JI,2d]
                    
                    fw_lpis = tf.reshape(fw_lpis_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
                    bw_lpis = tf.reshape(bw_lpis_flat.h,[N,M,d])
                    #fw_lpis = tf.Print(fw_lpis, [tf.shape(fw_lpis)], message="fw_lpis debug info", summarize=100)
                    #bw_lpis = tf.Print(bw_lpis, [tf.shape(bw_lpis)], message="bw_lpis debug info", summarize=100)
                    
                    lpis = tf.concat([fw_lpis,bw_lpis],2) # [N,M,2d]
                    #lpis = tf.Print(lpis, [tf.shape(lpis)], message="lpis debug info", summarize=100)
                
            if config.wd is not None: # l2 weight decay for the reader
                add_wd(config.wd)

        # all rnn output
        # hq -> [N,JQ,2d]

        # hat -> [N,M,JXA,2d]
        # had -> [N,M,JD,2d]
        # hwhen -> [N,M,JT,2d]
        # hwhere -> [N,M,JG,2d]
        # hpts -> [N,M,JI,JXP,2d]
        # hpis -> [N,M,JI,2d]

        # hchoices -> [N,4,JA,2d]

        # last states:
        # lq -> [N,2d]

        # lat -> [N,M,2d]
        # lad -> [N,M,2d]
        # lwhen -> [N,M,2d]
        # lwhere -> [N,M,2d]
        # lpts -> [N,M,JI,2d]
        # lpis -> [N,M,2d]

        # lchoices -> [N,4,2d]
        #hq = tf.Print(hq, [tf.shape(hq)], message="hq debug info", summarize=100)
        #hchoices = tf.Print(hchoices, [tf.shape(hchoices)], message="hchoices debug info", summarize=100)

        #hat = tf.Print(hat, [tf.shape(hat)], message="hat debug info", summarize=100)
        #had = tf.Print(had, [tf.shape(had)], message="had debug info", summarize=100)
        #hwhen = tf.Print(hwhen, [tf.shape(hwhen)], message="hwhen debug info", summarize=100)
        #hwhere = tf.Print(hwhere, [tf.shape(hwhere)], message="hwhere debug info", summarize=100)
        #hpts = tf.Print(hpts, [tf.shape(hpts)], message="hpts debug info", summarize=100)
        #hpis = tf.Print(hpis, [tf.shape(hpis)], message="hpis debug info", summarize=100)

        with tf.variable_scope("attention"):
            if config.use_attention:
                if config.attention_mode == 2:
                    K = 6
                    had = tf.pad(had, paddings=[[0, 0], [0, 0], [0, 40 - JD], [0, 0]], mode="CONSTANT", name="had_pad") 
                    #had = tf.Print(had, [tf.shape(had)], message="had debug info", summarize=100)
                    had_reduce = tf.reshape(had,[N,-1,10,4,2*d])  #[N,M,JD,2d]->[N,M,JD/4,4,2d]
                    had_reduce = tf.reduce_mean(had_reduce,3) #[N,M,JD/4,2d] #JD=40,JD/4=10
                    

                    hpts_reduce = tf.reduce_mean(hpts,3) #[N,M,JI,JXP,2d] -> [N,M,JI,2d]  #M=8,JI=10,JXP=10
                    
                    JMAX = tf.reduce_max([JXA, tf.shape(had_reduce)[-2], JT, JG, JI])

                    hat_pad = tf.pad(hat, paddings=[[0, 0], [0, 0], [0, JMAX-JXA], [0, 0]], mode="CONSTANT", name="hat_pad") 
                    
                    #hat_pad = tf.reshape(hat_pad, [N, M, -1, 2*d]) # need to do this to let tf know the shape?

                    had_pad = tf.pad(had_reduce, paddings=[[0, 0], [0, 0], [0, JMAX-tf.shape(had_reduce)[-2]], [0, 0]], mode="CONSTANT", name="had_pad")

                    hwhen_pad = tf.pad(hwhen, paddings=[[0, 0], [0, 0], [0, JMAX-JT], [0, 0]], mode="CONSTANT", name="hwhen_pad")

                    hwhere_pad = tf.pad(hwhere, paddings=[[0, 0], [0, 0], [0, JMAX-JG], [0, 0]], mode="CONSTANT", name="hwhere_pad")

                    hpis_pad = tf.pad(hpis, paddings=[[0, 0], [0, 0], [0, JMAX-JI], [0, 0]], mode="CONSTANT", name="hpis_pad")

                    hpts_pad = tf.pad(hpts_reduce, paddings=[[0, 0], [0, 0], [0, JMAX-JI], [0, 0]], mode="CONSTANT", name="hpts_pad")


                    hall = tf.stack([hat_pad, had_pad, hwhen_pad, hwhere_pad, hpts_pad, hpis_pad], axis=1) # [N, K, M, JMAX, 2d]
                    #hall = tf.Print(hall, [tf.shape(hall)], message="hall debug info", summarize=100)
                    hall = tf.reshape(hall, [N, K, -1, 2*d]) # [N, K, M, JMAX, 2d] -> [N,K,M*JMAX,2d]
                    #hall = tf.Print(hall, [tf.shape(hall)], message="hall debug info", summarize=100)

                    T = tf.shape(hall)[-2]

                    hall_aug = tf.tile(tf.expand_dims(hall,3),[1,1,1,T,1]) #[N,K,T,T,2d]
                    #hall_aug = tf.Print(hall_aug, [tf.shape(hall_aug)], message="hall_aug debug info", summarize=100)
                    hall_aug_t = tf.tile(tf.expand_dims(hall,2),[1,1,T,1,1]) #[N,K,T,T,2d] 
                    #hall_aug_t = tf.Print(hall_aug_t, [tf.shape(hall_aug_t)], message="hall_aug_t debug info", summarize=100)
                    simi = similarity(hall_aug,hall_aug_t) #[N,K,T,T,4d]
                    #simi = tf.Print(simi, [tf.shape(simi)], message="simi debug info", summarize=100)

                    lq_aug = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(lq, 1), 1),1), [1, K, T, T, 1]) #[N,2d] -> [N,K,T,T,2d]
                    #lq_aug = tf.Print(lq_aug, [tf.shape(lq_aug)], message="lq_aug debug info", summarize=100)
                    C_logits = linear(simi,2*d,scope="c_linear1",add_tanh=False,have_bias=False) + lq_aug #[N,K,T,T,2d]
                    #C_logits = tf.Print(C_logits, [tf.shape(C_logits)], message="C_logits debug info", summarize=100)
                    C_logits = linear(C_logits,1,scope="c_linear2",add_tanh=False,have_bias=False) #[N,K,T,T,1]
                    #C_logits = tf.Print(C_logits, [tf.shape(C_logits)], message="C_logits debug info", summarize=100)
                    C_logits = tf.squeeze(C_logits,-1) #[N,K,T,T]
                    #C_logits = tf.Print(C_logits, [tf.shape(C_logits)], message="C_logits debug info", summarize=100)
                    C = tf.tanh(tf.reduce_sum(C_logits, 1)) #[N,T,T]


                    # get new context
                    hall_aug_for_F = tf.tile(tf.expand_dims(hall, 3), [1, 1, 1, T, 1]) #[N,K,T,2D] -> [N, K, T, T, 2d]
                    #hall_aug_for_F = tf.Print(hall_aug_for_F, [tf.shape(hall_aug_for_F)], message="hall_aug_for_F debug info", summarize=100)
                    C_aug = tf.expand_dims(tf.tile(tf.expand_dims(C, 1), [1, K, 1, 1]), -1)# [N,T,T] -> [N, K, T, T, 1]
                    #C_aug = tf.Print(C_aug, [tf.shape(C_aug)], message="C_aug debug info", summarize=100)
                    F = tf.reduce_sum(hall_aug_for_F*C_aug, -2) #[N, K, T, 2d]
                    #F = tf.Print(F, [tf.shape(F)], message="F debug info", summarize=100)

                    #h_hat:[N,2d], q_hat:[N,2d]
                    h_hat,q_hat = attention2(hinfo=F, hq=hq, C=C, scope="cross_attention")

                if config.attention_mode == 1:
                    #album_title
                    A_mul_F_at,B_at,A_mul_Q_at,B_q_at = attention1(hat, hq, scope="album_title", dim_last = 2*d) #JXA=10
                    #album_description
                    had = tf.pad(had, paddings=[[0, 0], [0, 0], [0, 40 - JD], [0, 0]], mode="CONSTANT", name="had_pad") 
                    had_reduce = tf.reshape(had,[N,-1,10,4,2*d])  #[N,M,JD,2d]->[N,M,JD/4,4,2d]
                    had_reduce = tf.reduce_mean(had_reduce,3) #[N,M,JD/4,2d] #JD=40,JD/4=10

                    A_mul_F_ad,B_ad,A_mul_Q_ad,B_q_ad = attention1(had_reduce, hq, scope="album_description", dim_last = 2*d) #JD=40,JD/4=10
                    #when
                    A_mul_F_when,B_when,A_mul_Q_when,B_q_when = attention1(hwhen, hq, scope="when", dim_last = 2*d) #JT=4
                    #where
                    A_mul_F_where,B_where,A_mul_Q_where,B_q_where = attention1(hwhere, hq, scope="where", dim_last = 2*d) #JG=4
                    #photo_title
                    hpts_reduce = tf.reduce_mean(hpts,3) #[N,M,JI,JXP,2d] -> [N,M,JI,2d]
                    A_mul_F_pts,B_pts,A_mul_Q_pts,B_q_pts = attention1(hpts_reduce, hq, scope="photo_title", dim_last = 2*d) #M=8,JI=10,JXP=10
                    #photo
                    A_mul_F_pis,B_pis,A_mul_Q_pis,B_q_pis = attention1(hpis, hq, scope="photo", dim_last = 2*d) #JI=10

                    
                    #A_mul_F_at = tf.Print(A_mul_F_at, [tf.shape(A_mul_F_at)], message="A_mul_F_at debug info", summarize=100)
                    #A_mul_F_ad = tf.Print(A_mul_F_ad, [tf.shape(A_mul_F_ad)], message="A_mul_F_ad debug info", summarize=100)
                    #A_mul_F_when = tf.Print(A_mul_F_when, [tf.shape(A_mul_F_when)], message="A_mul_F_when debug info", summarize=100)
                    #A_mul_F_where = tf.Print(A_mul_F_where, [tf.shape(A_mul_F_where)], message="A_mul_F_where debug info", summarize=100)
                    #A_mul_F_pts = tf.Print(A_mul_F_pts, [tf.shape(A_mul_F_pts)], message="A_mul_F_pts debug info", summarize=100)
                    #A_mul_F_pis = tf.Print(A_mul_F_pis, [tf.shape(A_mul_F_pis)], message="A_mul_F_pis debug info", summarize=100)

                    A_mul_F = tf.stack([A_mul_F_at,A_mul_F_ad,A_mul_F_when,A_mul_F_where,A_mul_F_pts,A_mul_F_pis],axis=2) #[N,2d,6]
                    #A_mul_F = tf.Print(A_mul_F, [tf.shape(A_mul_F)], message="A_mul_F debug info", summarize=100)
                    B = tf.stack([B_at,B_ad,B_when,B_where,B_pts,B_pis],axis=1) #[N,6]
                    B = tf.nn.softmax(B,axis=-1)
                    #B = tf.Print(B, [tf.shape(B)], message="B debug info", summarize=100)
                    h_hat = tf.matmul(A_mul_F,tf.expand_dims(B,2)) #[N,2d,1]
                    h_hat = tf.squeeze(h_hat,-1) #[N,2d]
                    
                    A_mul_Q = tf.stack([A_mul_Q_at,A_mul_Q_ad,A_mul_Q_when,A_mul_Q_where,A_mul_Q_pts,A_mul_Q_pis],axis=2) #[N,2d,6]
                    #A_mul_Q = tf.Print(A_mul_Q, [tf.shape(A_mul_Q)], message="A_mul_Q debug info", summarize=100)       
                    B_q = tf.stack([B_q_at,B_q_ad,B_q_when,B_q_where,B_q_pts,B_q_pis],axis=1) #[N,6]
                    B_q = tf.nn.softmax(B_q,axis=-1)
                    #B_q = tf.Print(B_q, [tf.shape(B_q)], message="B_q debug info", summarize=100)
                    q_hat = tf.matmul(A_mul_Q,tf.expand_dims(B_q,2)) #[N,2d,1]
                    q_hat = tf.squeeze(q_hat,-1) #[N,2d]
               
            else:
                # use last lstm output (last hidden state)
                # outputs_fw[k,X_len[k]-1] == states_fw.h[k]
                # at_len -> [N,M]
                g0at = lat #[N,M,2d]
                g0ad = lad # [N,M,2d]
                g0when = lwhen
                g0where = lwhere
                g0pts = tf.reduce_mean(lpts,2) #[N,M,JI,2d] -> [N,M,2d]
                g0pis = lpis

                # album level attention
                g1at = tf.reduce_mean(g0at,1) # [N,2d]
                g1ad = tf.reduce_mean(g0ad,1)
                g1when = tf.reduce_mean(g0when,1)
                g1where = tf.reduce_mean(g0where,1)
                g1pts = tf.reduce_mean(g0pts,1)
                g1pis = tf.reduce_mean(g0pis,1)

        if config.use_attention:
            g1_all = h_hat #[N,2d]
            #g1_all = tf.Print(g1_all, [tf.shape(g1_all)], message="g1_all debug info", summarize=100)
        else:
            g1 = tf.stack([g1at,g1ad,g1when,g1where,g1pts,g1pis],axis=1)    #[N,6,2d] [64,6,200]
            #g1 = tf.Print(g1, [tf.shape(g1)], message="g1 debug info", summarize=100)
            g1_all = tf.reduce_mean(g1,1) #[N,2d] [64,200]
            #g1_all = tf.Print(g1_all, [tf.shape(g1_all)], message="g1_all debug info", summarize=100)
      
        with tf.variable_scope("choices_emb"):
            # embed the choices

            gchoices = lchoices #[N,4,2d] # last LSTM state for each choice

        with tf.variable_scope("question_emb"):
            # hq -> [N,JQ,2d]
            # gp -> [N,2d]
            if config.use_attention:
                gq = q_hat #[N,2d]        
            else:
                gq = lq # this is the last hidden state of each question # [N,2d]


        # the modeling layer
        with tf.variable_scope("output"):
            
            # g1_all [N,2d] # this could be viewed as an answer representation
            # together with the choices_emb and question_emb, 
            # we do a single layer multi class classification

            # tile g1_all [N,2d] -> [N,1,2d] to concat with gchoices
            # [N,4,2d]
            g1_a_t = tf.tile(tf.expand_dims(g1_all,1),[1,self.num_choice,1])

            # tile gq for all choices
            gq = tf.tile(tf.expand_dims(gq,1),[1,self.num_choice,1]) # [N,4,2d]


            # [N,4,2d] -> [N*4,1]
            # TODO: consider different similarity matrix
            
            logits = linear(tf.concat([gq,g1_a_t,gchoices,g1_a_t*gchoices,gq*gchoices],2),output_size=1,scope="choicelogits")
            

            logits = tf.squeeze(logits,2) # [N,4,1] -> [N,4]
            yp = tf.nn.softmax(logits) # [N,4]

            # for loss and forward
            self.logits = logits
            self.yp = yp

    def build_loss(self):
        # logits -> [N,4]
        # y -> [N,4]
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=tf.cast(self.y,"float")) # [N] # softmax cross entropy loss.
        #
        losses = tf.reduce_mean(losses) # scalar, avg loss of the whole batch

        tf.add_to_collection("losses",losses)

        # there might be l2 weight loss in some layer
        self.loss = tf.add_n(tf.get_collection("losses"),name="total_losses")


    # givng a batch of data, construct the feed dict
    def get_feed_dict(self,batch,is_train=False):
        assert isinstance(batch,Dataset)
        # get the cap for each kind of step first
        config = self.config
        N = config.batch_size
        #N = len(batch.data['q'])
        M = config.max_num_albums
        #JX = config.max_sent_title_size
        JXA = config.max_sent_album_title_size
        JXP = config.max_sent_photo_title_size
        JD = config.max_sent_des_size
        JQ = config.max_question_size
        JI = config.max_num_photos
        JT = config.max_when_size
        JG = config.max_where_size
        JA = config.max_answer_size

        VW = config.word_vocab_size
        VC = config.char_vocab_size
        d = config.hidden_size
        W = config.max_word_size

        # This could make training faster
        # so each minibatch 's max length is different
        
        new_JXA = max(len(title) for sample in batch.data['album_title'] for title in sample)
        new_JXP = max([len(title) for sample in batch.data['photo_titles'] for album in sample for title in album]+[0])
        if new_JXA == 0: # empty??
            new_JXA = 1
        if new_JXP == 0: # empty??
            new_JXP = 1
        #JX = min(JX,new_JX) # so JX should be the longest sentence  in the batch, but may not be the longest in the whole dataset
        JXA = min(JXA,new_JXA)
        JXP = min(JXP,new_JXP)

        new_JD = max(len(des) for sample in batch.data['album_description'] for des in sample)
        if new_JD == 0: # empty??
            new_JD = 1
        JD = min(JD,new_JD)

        new_JG = max(len(where) for sample in batch.data['where'] for where in sample)
        if new_JG == 0: # could be empty
            new_JG = 1
        JG = min(JG,new_JG)

        new_JT = max(len(when) for sample in batch.data['when'] for when in sample)
        if new_JT == 0: # empty??
            new_JT = 1
        JT = min(JT,new_JT)

        new_JI = max(len(album) for sample in batch.data['photo_ids'] for album in sample)
        if new_JI == 0: # empty??
            new_JI = 1
        JI = min(JI,new_JI)

        new_JQ = max(len(ques) for ques in batch.data['q'])
        if(new_JQ == 0):
            new_JQ = 1
        JQ = min(JQ,new_JQ)


        new_M = max(len(onesample) for onesample in batch.data['album_title'])
        if(new_M == 0):
            new_M = 1
        M = min(M,new_M)

        feed_dict = {}

        # initial all the placeholder
        # all words initial is 0 , means -NULL- token
        at = np.zeros([N,M,JXA],dtype='int32')
        at_c = np.zeros([N,M,JXA,W],dtype="int32")
        at_mask = np.zeros([N,M,JXA],dtype="bool")

        ad = np.zeros([N,M,JD],dtype='int32')
        ad_c = np.zeros([N,M,JD,W],dtype="int32")
        ad_mask = np.zeros([N,M,JD],dtype="bool")

        when = np.zeros([N,M,JT],dtype='int32')
        when_c = np.zeros([N,M,JT,W],dtype="int32")
        when_mask = np.zeros([N,M,JT],dtype="bool")

        where = np.zeros([N,M,JG],dtype='int32')
        where_c = np.zeros([N,M,JG,W],dtype="int32")
        where_mask = np.zeros([N,M,JG],dtype="bool")

        pts = np.zeros([N,M,JI,JXP],dtype="int32")
        pts_c = np.zeros([N,M,JI,JXP,W],dtype="int32")
        pts_mask = np.zeros([N,M,JI,JXP],dtype="bool")

        pis = np.zeros([N,M,JI],dtype='int32')
        pis_mask = np.zeros([N,M,JI],dtype="bool")

        q = np.zeros([N,JQ],dtype='int32')
        q_c = np.zeros([N,JQ,W],dtype="int32")
        q_mask = np.zeros([N,JQ],dtype="bool")

        choices = np.zeros([N,self.num_choice,JA],dtype='int32')
        choices_c = np.zeros([N,self.num_choice,JA,W],dtype="int32")
        choices_mask = np.zeros([N,self.num_choice,JA],dtype="bool")


        # link the feed_dict
        feed_dict[self.at] = at
        feed_dict[self.at_c] = at_c
        feed_dict[self.at_mask] = at_mask

        feed_dict[self.ad] = ad
        feed_dict[self.ad_c] = ad_c
        feed_dict[self.ad_mask] = ad_mask

        feed_dict[self.when] = when
        feed_dict[self.when_c] = when_c
        feed_dict[self.when_mask] = when_mask

        feed_dict[self.where] = where
        feed_dict[self.where_c] = where_c
        feed_dict[self.where_mask] = where_mask

        feed_dict[self.pts] = pts
        feed_dict[self.pts_c] = pts_c
        feed_dict[self.pts_mask] = pts_mask

        feed_dict[self.pis] = pis
        feed_dict[self.pis_mask] = pis_mask

        feed_dict[self.q] = q
        feed_dict[self.q_c] = q_c
        feed_dict[self.q_mask] = q_mask

        feed_dict[self.choices] = choices
        feed_dict[self.choices_c] = choices_c
        feed_dict[self.choices_mask] = choices_mask

        feed_dict[self.is_train] = is_train

        # image feat mat and word mat
        feed_dict[self.image_emb_mat] = batch.data['pidx2feat']
        feed_dict[self.existing_emb_mat] = batch.shared['existing_emb_mat']


        # question and choices
        Q = batch.data['q']
        Q_c = batch.data['cq'] 

        C = deepcopy(batch.data['cs']) # for the choice, since we will add correct answer into it, we copy so it won't affect other batch
        C_c = deepcopy(batch.data['ccs'])

        # data
        AT = batch.data['album_title']
        AT_c = batch.data['album_title_c']
        AD = batch.data['album_description']
        AD_c = batch.data['album_description_c']
        WHERE = batch.data['where']
        WHERE_c = batch.data['where_c']
        WHEN = batch.data['when']
        WHEN_c = batch.data['when_c']

        PT = batch.data['photo_titles']
        PT_c = batch.data['photo_titles_c']

        PI = batch.data['photo_idxs']

        # for training, one of the y will be in the choices
        # only training feed the y
        if is_train:
            Y = batch.data['y']
            Y_c = batch.data['cy']

            y = np.zeros([N,self.num_choice],dtype="bool")
            feed_dict[self.y] = y

            # decide the index of correct choice first, we randomly decide it
            correctIndex = np.random.choice(self.num_choice,N) # get a array of size [N]


            #for i in range(N): # some batch will be smaller
            for i in range(len(batch.data['y'])):
                y[i,correctIndex[i]] = True
                # put the answer into the choices
                assert len(C[i]) == (self.num_choice - 1)
                C[i].insert(correctIndex[i],Y[i])
                C_c[i].insert(correctIndex[i],Y_c[i])
                assert len(batch.data['cs'][i]) == (self.num_choice - 1)

        else:
            # for testing, put the answer into the original idx if there is any
            if("y" in batch.data and "cy"in batch.data and "yidx" in batch.data):
                Y = batch.data['y']
                Y_c = batch.data['cy']
                Y_idx = batch.data['yidx']
                #for i in range(N): # some batch will be smaller
                for i in range(len(batch.data['y'])):
                    #print i,len(C[i])
                    assert len(C[i]) == (self.num_choice - 1), ("C[i] len:%s,%s,Y:%s"%(len(C[i]),C[i],Y[i]))
                    C[i].insert(Y_idx[i],Y[i])
                    C_c[i].insert(Y_idx[i],Y_c[i])
                    


        # the photo idx is simple

        for i,pi in enumerate(PI):
            # one batch
            for j,pij in enumerate(pi):
                # one album
                if j == config.max_num_albums:
                    break
                for k,pijk in enumerate(pij):
                    if k == config.max_num_photos:
                        break
                    #print pijk
                    assert isinstance(pijk,int)
                    pis[i,j,k] = pijk
                    pis_mask[i,j,k] = True



        def get_word(word):
            d = batch.shared['word2idx'] # this is for the word not in glove
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            # the word in glove
            
            d2 = batch.shared['existing_word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d2:
                    return d2[each] + len(d) # all idx + len(the word to train)
            return 1 # 1 is the -UNK-

        def get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        # for all the text, get each word's index.
        # album title
        for i, ati in enumerate(AT): # batch_sizes
            # one batch
            for j,atij in enumerate(ati):
                # one album
                if j == config.max_num_albums:
                    break
                for k,atijk in enumerate(atij):
                    # each word
                    if k == config.max_sent_album_title_size:
                        break
                    wordIdx = get_word(atijk)
                    at[i,j,k] = wordIdx
                    at_mask[i,j,k] = True

        for i, cati in enumerate(AT_c):
            # one batch
            for j, catij in enumerate(cati):
                if j == config.max_num_albums:
                    break
                for k, catijk in enumerate(catij):
                    # each word
                    if k == config.max_sent_album_title_size:
                        break
                    for l,catijkl in enumerate(catijk):
                        if l == config.max_word_size:
                            break
                        at_c[i,j,k,l] = get_char(catijkl)



        # album description
        for i, adi in enumerate(AD): # batch_sizes
            # one batch
            for j,adij in enumerate(adi):
                # one album
                if j == config.max_num_albums:
                    break
                for k,adijk in enumerate(adij):
                    # each word
                    if k == config.max_sent_des_size:
                        break
                    wordIdx = get_word(adijk)
                    ad[i,j,k] = wordIdx
                    ad_mask[i,j,k] = True

        for i, cadi in enumerate(AD_c):
            # one batch
            for j, cadij in enumerate(cadi):
                if j == config.max_num_albums:
                    break
                for k, cadijk in enumerate(cadij):
                    # each word
                    if k == config.max_sent_des_size:
                        break
                    for l,cadijkl in enumerate(cadijk):
                        if l == config.max_word_size:
                            break
                        ad_c[i,j,k,l] = get_char(cadijkl)




        # album when
        for i, wi in enumerate(WHEN): # batch_sizes
            # one batch
            for j,wij in enumerate(wi):
                # one album
                if j == config.max_num_albums:
                    break
                for k,wijk in enumerate(wij):
                    # each word
                    if k == config.max_when_size:
                        break
                    wordIdx = get_word(wijk)
                    when[i,j,k] = wordIdx
                    when_mask[i,j,k] = True

        for i, cwi in enumerate(WHEN_c):
            # one batch
            for j, cwij in enumerate(cwi):
                if j == config.max_num_albums:
                    break
                for k, cwijk in enumerate(cwij):
                    # each word
                    if k == config.max_when_size:
                        break
                    for l,cwijkl in enumerate(cwijk):
                        if l == config.max_word_size:
                            break
                        when_c[i,j,k,l] = get_char(cwijkl)

        # album where
        for i, wi in enumerate(WHERE): # batch_sizes
            # one batch
            for j,wij in enumerate(wi):
                # one album
                if j == config.max_num_albums:
                    break
                for k,wijk in enumerate(wij):
                    # each word
                    if k == config.max_where_size:
                        break
                    wordIdx = get_word(wijk)
                    where[i,j,k] = wordIdx
                    where_mask[i,j,k] = True

        for i, cwi in enumerate(WHERE_c):
            # one batch
            for j, cwij in enumerate(cwi):
                if j == config.max_num_albums:
                    break
                for k, cwijk in enumerate(cwij):
                    # each word
                    if k == config.max_where_size:
                        break
                    for l,cwijkl in enumerate(cwijk):
                        if l == config.max_word_size:
                            break
                        where_c[i,j,k,l] = get_char(cwijkl)


        # photo title
        for i, pti in enumerate(PT): # batch_sizes
            # one batch
            for j,ptij in enumerate(pti):
                # one album
                if j == config.max_num_albums:
                    break
                for k,ptijk in enumerate(ptij):
                    # each photo
                    if k == config.max_num_photos:
                        break
                    for l,ptijkl in enumerate(ptijk):
                        if l == config.max_sent_photo_title_size:
                            break
                        # each word
                        wordIdx = get_word(ptijkl)
                        pts[i,j,k,l] = wordIdx
                        pts_mask[i,j,k,l] = True
        for i, pti in enumerate(PT_c): # batch_sizes
            # one batch
            for j,ptij in enumerate(pti):
                # one album
                if j == config.max_num_albums:
                    break
                for k,ptijk in enumerate(ptij):
                    # each photo
                    if k == config.max_num_photos:
                        break
                    for l,ptijkl in enumerate(ptijk):
                        if l == config.max_sent_photo_title_size:
                            break
                        # each word
                        for o, ptijklo in enumerate(ptijkl):
                            # each char
                            if o == config.max_word_size:
                                break
                            pts_c[i,j,k,l,o] = get_char(ptijklo)



        # answer choices

        for i,ci in enumerate(C):
            # one batch
            assert len(ci) == self.num_choice
            for j,cij in enumerate(ci):
                # one answer
                for k,cijk in enumerate(cij):
                    # one word
                    if k == config.max_answer_size:
                        break
                    wordIdx = get_word(cijk)
                    choices[i,j,k] = wordIdx
                    choices_mask[i,j,k] = True
        for i,ci in enumerate(C_c):
            # one batch
            assert len(ci) == self.num_choice, (len(ci))
            for j,cij in enumerate(ci):
                # one answer
                for k,cijk in enumerate(cij):
                    # one word
                    if k == config.max_answer_size:
                        break
                    for l,cijkl in enumerate(cijk):
                        if l == config.max_word_size:
                            break
                        choices_c[i,j,k,l] = get_char(cijkl)


        # loa the question
        # no limiting on the question word length
        for i, qi in enumerate(Q):
            # one batch
            for j, qij in enumerate(qi):
                q[i, j] = get_word(qij)
                q_mask[i, j] = True

        # load the question char
        for i, cqi in enumerate(Q_c):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    
                    if k == config.max_word_size:
                        break
                    q_c[i, j, k] = get_char(cqijk)
        return feed_dict


