import torch.nn as nn
from transformers import BertModel
import torch
from transformers import AutoTokenizer, AutoModel
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from transformers import *
import pandas as pd
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import GRU, TimeDistributed
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense
from keras.layers import Embedding, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant
from keras import backend as K

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="E:/transformer_file/")
model_config = BertConfig.from_pretrained(MODEL_PATH)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
embedding = AutoModel.from_pretrained(MODEL_PATH) 
model = BertModel.from_pretrained(MODEL_PATH, config = model_config)

text = []
datasets_ch = ['Weibo-16', 'Weibo-16-original', 'Weibo-20', 'Weibo-20-temporal']
dataset_dir = '/home/alissa77/WWW2021/code/preprocess/data'


for dataset in datasets_ch : 
    data_dir = os.path.join(dataset_dir, dataset)  
    split_datasets = [pd.read_pickle(os.path.join(data_dir, '{}.pkl'.format(t))) for t in ['train', 'val', 'test']] 
    split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))
    for t, pieces in split_datasets.items(): 
        text.append(pieces['content_words'])
            
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0) 
        outputs = model(input_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        print("inputs :",input_ids)
        print()
        print("單字向量",sequence_output.shape)    ## 字向量
        print("句子向量",pooled_output.shape)      ## 句向量


# Bert-BiGRU-Classifier
class BERTmodel(nn.Module):
    def __init__(self, max_sequence_length, embedding_matrix, emotion_dim=0, 
                 category_num=3, hidden_units=32, l2_param=0.01, lr_param=0.001):
        
        self.max_sequence_length = max_sequence_length
        self.embedding = BertModel.from_pretrained('bert-base-chinese')
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=768,
            dropout=0.3,
            num_layers=5,
            bidirectional=True,
            batch_first=True,
        )
        self.emotion_dim = emotion_dim
        self.hidden_units = hidden_units
        self.category_num = category_num
        self.l2_param = l2_param
        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=lr_param, beta_1=0.8), metrics=['accuracy'])
        
        #self.gru可以和下面參數合併使用嗎？


    #修改model layer  
    def build(self, tokens, masks=None):
        semantic_input = Input(shape=(self.max_sequence_length,), name='word-embedding-input')
        semantic_emb = embedded, _ = self.embedding(tokens, attention_mask=masks)
        cls_vector = embedded[:, 0, :]
        cls_vector = cls_vector.view(-1, 1, 768)
        gru = Bidirectional(GRU(self.hidden_units, return_sequences=True))(semantic_emb)
        avg_pool = GlobalAveragePooling1D()(gru)
        
        # emotion_dim
        emotion_input = Input(shape=(self.emotion_dim,), name='emotion-input')
        emotion_enhanced = Concatenate()([avg_pool, emotion_input])
        dense = Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param))(emotion_enhanced)
        output = Dense(self.category_num, activation='softmax',
                        kernel_regularizer=l2(self.l2_param))(dense)
        
        model = Model(inputs=[semantic_input, emotion_input], outputs=output)

        return model