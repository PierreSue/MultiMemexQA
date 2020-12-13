# coding=utf-8
# Team: CUDA out of memory
# Team Meambers: Feng-Guang Su, Yiwei Qin, Rouxin Huang, Jeremiah Cheng
# Class: 11785 Introduction to Deep Learning at CMU
# Revised From https://github.com/JunweiLiang/MemexQA_StarterCode

# Dataset download
	memexqa_dataset_v1.1/
	├── album_info.json   # album data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
	├── glove.6B.100d.txt # word vectors for baselines:  http://nlp.stanford.edu/data/glove.6B.zip
	├── photos_inception_resnet_v2_l2norm.npz # photo features: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
	├── qas.json # QA data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
	└── test_question.ids # testset Id: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids

# LSTM_based
    Preprocess:
    python3 LSTM_based/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz memexqa_dataset_v1.1/glove.6B.100d.txt prepro_BERT --use_BERT
	Train: 
    python3 LSTM_based/main.py prepro_BERT/ bert_CL_FVTA --modelname simple_lstm --keep_prob 0.7 --is_train --use_attention  --use_char --image_linear --batch_size 8
    Test:
    python3 LSTM_based/main.py prepro_BERT/ bert_CL_FVTA --modelname simple_lstm --keep_prob 0.7 --is_test --use_attention  --use_char --image_linear --batch_size 8  --load_best

# Self_Attention_based
    Preprocess:
    python3 Self_Attention_based/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz prepro_BERT_SA --word_based
    Train:
    python3 Self_Attention_based/train.py --workers 8 --batchSize 32 --niter 100 --inpf ./prepro_BERT_SA/ --outf ./outputs/BERT_WL_SA --cuda --gpu_id 0
    Test:
    python3 Self_Attention_based/test.py --workers 8 --batchSize 32 --inpf ./prepro_BERT_SA/ --outf ./outputs/BERT_WL_SA --cuda --gpu_id 0
