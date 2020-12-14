# Introduction
This is the final project in the class "11785 Introduction to Deep Learning(FALL2020)" at CMU. We are the team "CUDA out of memory". 

With the development of photography technology, one may create several personal albums accumulating thousands of photos and many hours of videos to capture important events in their life. While that is a blessing in this new age of technology, it can lead to a problem of disorganization and information overload. Previous work attempts to resolve this problem with an automated way of processing and organizing photos, by proposing a new Visual Question Answering task named MemexQA. MemexQA works by taking the input of the user, the input being a question and a series of pictures, and responds with an answer to that question and a separate series of photos to justify the answer. A natural way to recall the past are by questions and photographs, and MemexQA eloquently combines the two to organize and label photographs, whether it be for the user to recall the past or to organize photographs into separate albums. Our objective for our project is to follow up on the Visual Question Answering task by proposing a new neural network architecture that attains a higher performance on the MemexQA dataset.

Based on the Visual Question Answering task, MemexQA, proposed by ["Focal Visual-Text Attention for Memex Question Answering"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8603827), we set out to test several alternative models to eventually compile our results to theorize a final model to gain a better overall performance compared to the originally proposed model described by [the paper author](https://github.com/JunweiLiang/MemexQA_StarterCode). Overall we were able to produce a final model that beat the original baseline by roughly 10% in accuracy. 

# Dataset download
	memexqa_dataset_v1.1/
	├── album_info.json   # album data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
	├── glove.6B.100d.txt # word vectors for baselines:  http://nlp.stanford.edu/data/glove.6B.zip
	├── photos_inception_resnet_v2_l2norm.npz # photo features: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
	├── qas.json # QA data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
	└── test_question.ids # testset Id: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids

# Collect Dataset
    mkdir memexqa_dataset_v1.1 
    cd memexqa_dataset_v1.1 
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
    wget http://nlp.stanford.edu/data/glove.6B.zip
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids
    unzip glove.6B.zip

# LSTM_based
Preprocess:
GloVe: \
    python3 LSTM_based/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz memexqa_dataset_v1.1/glove.6B.100d.txt prepro
BERT: \
    python3 LSTM_based/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz memexqa_dataset_v1.1/glove.6B.100d.txt prepro_BERT --use_BERT

Train:
GloVe-WL: \
    python3 LSTM_based/main.py prepro/ glove_WL --modelname simple_lstm --keep_prob 0.7 --is_train
GloVe-WL + FVTA: \
    python3 LSTM_based/main.py prepro/ glove_WL_FVTA --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 2 --is_train
GloVe-WL + Char + FVTA: \
    python3 LSTM_based/main.py prepro/ glove_WL_char_FVTA --modelname simple_lstm --keep_prob 0.7  --use_attention --attention_mode 2 --use_char --is_train
GloVe-WL + Char + FVTA-S: \
    python3 LSTM_based/main.py prepro/ glove_WL_char_FVTA_S --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 1 --use_char --is_train
BERT-WL: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL --modelname simple_lstm --keep_prob 0.7 --is_train
BERT-WL + FVTA: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL_FVTA --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 2 --is_train
BERT-WL + Char + FVTA: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL_char_FVTA --modelname simple_lstm --keep_prob 0.7  --use_attention --attention_mode 2 --use_char --is_train
BERT-WL + Char + FVTA-S: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL_char_FVTA_S --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 1 --use_char --is_train

Test:
GloVe-WL: \
    python3 LSTM_based/main.py prepro/ glove_WL --modelname simple_lstm --keep_prob 0.7 --is_test
GloVe-WL + FVTA: \
    python3 LSTM_based/main.py prepro/ glove_WL_FVTA --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 2 --is_test
GloVe-WL + Char + FVTA: \
    python3 LSTM_based/main.py prepro/ glove_WL_char_FVTA --modelname simple_lstm --keep_prob 0.7  --use_attention --attention_mode 2 --use_char --is_test
GloVe-WL + Char + FVTA-S: \
    python3 LSTM_based/main.py prepro/ glove_WL_char_FVTA_S --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 1 --use_char --is_test
BERT-WL: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL --modelname simple_lstm --keep_prob 0.7 --is_test
BERT-WL + FVTA: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL_FVTA --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 2 --is_test
BERT-WL + Char + FVTA: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL_char_FVTA --modelname simple_lstm --keep_prob 0.7  --use_attention --attention_mode 2 --use_char --is_test
BERT-WL + Char + FVTA-S: \
    python3 LSTM_based/main.py prepro_BERT/ bert_WL_char_FVTA_S --modelname simple_lstm --keep_prob 0.7 --use_attention --attention_mode 1 --use_char --is_test --load_best

# Self_Attention_based
Preprocess: \
    python3 Self_Attention_based/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz prepro_BERT_SA --word_based
Train: \
    python3 Self_Attention_based/train.py --workers 8 --batchSize 32 --niter 100 --inpf ./prepro_BERT_SA/ --outf ./outputs/BERT_WL_SA --cuda --gpu_id 0
Test: \
    python3 Self_Attention_based/test.py --workers 8 --batchSize 32 --inpf ./prepro_BERT_SA/ --outf ./outputs/BERT_WL_SA --cuda --gpu_id 0
