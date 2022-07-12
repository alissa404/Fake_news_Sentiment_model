import os
import json
from tqdm import tqdm
import time
import numpy as np
from keras.utils import to_categorical
save_dir = './no-sentiment'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

datasets_ch = ['Weibo-16','Weibo-20']

label2idx = {'fake': 0, 'real': 1, 'unverified': 2}

def get_labels_arr(pieces):
    labels = np.array([label2idx[p['label']] for p in pieces])
    return labels


for dataset in datasets_ch :
    data_dir = os.path.join('../../dataset', dataset)
    output_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    label_dir = os.path.join(output_dir, 'labels')
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r')) for t in ['train', 'val', 'test']]
    split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

    for t, pieces in split_datasets.items():
        labels_arr = get_labels_arr(pieces)
        print(labels_arr)
        print('{} dataset: got a {} label arr'.format(t, labels_arr.shape))
        np.save(os.path.join(label_dir, '{}_{}.npy'.format( t, labels_arr.shape)), labels_arr)