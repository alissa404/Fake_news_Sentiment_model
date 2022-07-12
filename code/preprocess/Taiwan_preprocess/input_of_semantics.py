import numpy as np
import json
import time
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('../../word-embedding')
from load_embeddings import load_embeddings

save_dir = '/home/alissa77/WWW2021/code/Taiwan_preprocess'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
datasets_ch = ['taiwan']

for dataset in datasets_ch :
    print('\n\n{} [{}]\tProcessing the dataset: {} {}\n'.format(
        '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dataset, '-'*20))

    MAX_NUM_WORDS = 6000
    if dataset in datasets_ch:
        embeddings_index = load_embeddings(
            language='Chinese', embeddings_file='/home/alissa77/WWW2021/word-embedding/sgns.weibo.bigram-char')
        CONTENT_WORDS = 100
        EMBEDDING_DIM = 300

    ################## load data ##########################
    data_dir = os.path.join(save_dir, dataset)
    output_dir = os.path.join(data_dir, 'semantics')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    split_datasets = [json.load(open(os.path.join(
        data_dir, '{}.json'.format(t)), 'r')) for t in ['train', 'test']]
    split_datasets = dict(zip(['train',  'test'], split_datasets))

    texts = []
    for t in ['train',  'test']:
        texts += [' '.join(p['content_words']) for p in split_datasets[t]]
    print('\n{}: {}, len(texts) = {}, \neg: texts[0] = {}\n'.format(
        dataset, sum([len(d) for d in split_datasets.values()]), len(texts), texts[0]))

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    content_arr = pad_sequences(sequences, maxlen=CONTENT_WORDS)
    print('Content Array: {}'.format(content_arr.shape))

    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.random.randn(num_words, EMBEDDING_DIM)
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Embedding Matrix: {}'.format(embedding_matrix.shape))

    np.save(os.path.join(output_dir, 'embedding_matrix_{}.npy'.format(
        embedding_matrix.shape)), embedding_matrix)

    a, b = len(split_datasets['train']), len(split_datasets['test'])
    arrs = [content_arr[:a], content_arr[:b]]
    for i, t in enumerate(['train', 'test']):
        np.save(os.path.join(output_dir, '{}_{}.npy'.format(
            t, arrs[i].shape)), arrs[i])