# BEEEEEST
import os
import json
from tqdm import tqdm
import time
import numpy as np
import sys
sys.path.append('../emotion')

# 改這邊
import BOSON_asa
extract_pkg = BOSON_asa
save_dir = '/home/alissa77/dataset/weibo'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

datasets_ch = ['Weibo-16', 'Weibo-20']
for dataset in datasets_ch :
    print('\n\n{} [{}]\tProcessing the dataset: {} {}\n'.format(
        '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dataset, '-'*20))
    data_dir = os.path.join('/home/alissa77/dataset', dataset)
    print(data_dir)
    output_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    emotion_dir_title = os.path.join(output_dir, 'emotions_title')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    emotion_dir = os.path.join(output_dir, 'emotions')
    if not os.path.exists(emotion_dir):
        os.mkdir(emotion_dir)

    split_datasets = [json.load(open(os.path.join(
        data_dir, '{}.json'.format(t)), 'r')) for t in ['train', 'val', 'test']]
    split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

    for t, pieces in split_datasets.items():
        # words cutting
        print('[{}]\tWords Cutting...'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        for p in tqdm(pieces):
            p['content_words'] = extract_pkg.cut_words_from_text(p['content'])
        for p in tqdm(pieces):
            p['title_words'] = extract_pkg.cut_words_from_text(p['title'])
        with open(os.path.join(output_dir, '{}.json'.format(t)), 'w') as f:
            json.dump(pieces, f, indent=4, ensure_ascii=False)   

        emotion_arr = [extract_pkg.extract_dual_emotion(p) for p in tqdm(pieces)]
        emotion_arr = np.array(emotion_arr)
        
        emotion_arr2 = [extract_pkg.extract_title_emotion(p) for p in tqdm(pieces)]
        emotion_arr2 = np.array(emotion_arr2)        
        print('{} dataset: got a {} emotion arr'.format(t, emotion_arr.shape))
        np.save(os.path.join(emotion_dir, '{}_{}.npy'.format( t, emotion_arr.shape)), emotion_arr)
        np.save(os.path.join(emotion_dir_title, '{}_{}.npy'.format( t, emotion_arr.shape)), emotion_arr2)
