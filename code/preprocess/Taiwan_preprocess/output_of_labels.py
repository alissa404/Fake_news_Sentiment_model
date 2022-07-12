import os
import json
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from fast_ml.model_development import train_valid_test_split

DATA_PATH = '/home/alissa77/practice/WWW2021_dEEFEND/dataset'

save_dir = './taiwan_data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
datasets_ch = ["taiwan"]        
label2idx = {'fake': 0,  'real': 1}

def get_labels_arr(pieces):
    labels = np.array([label2idx[p['label']] for p in pieces])
    return to_categorical(labels)

for dataset in datasets_ch :
    print('\n\n{} [{}]\tProcessing the dataset: {} {}\n'.format(
        '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dataset, '-'*20))

    data_dir = os.path.join(DATA_PATH, dataset)
    output_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    label_dir = os.path.join(output_dir, 'labels')
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r')) for t in ['train', 'test']]
    split_datasets = dict(zip(['train', 'test'], split_datasets))

    for t, pieces in split_datasets.items():
        # pieces 整份資料
        # for p in pieces:
        #     p['label'][p['label']==1]= 'fake'
        #     p['label'][p['label']==0 ] ='real'
        labels_arr = get_labels_arr(pieces)
        np.save(os.path.join(label_dir, '{}_{}.npy'.format(
            t, labels_arr.shape)), labels_arr)

######### 製作train, val, test.json ##########
#用glob去遞迴訪問每層資料夾裡面 news_content.json的檔案
# import glob 
# temp = []
# for i in ["fake", "real"] :
#     data_path = os.path.join(DATA_PATH ,"{}".format(i)) 
#     for j in os.listdir(data_path):
#         path = os.path.join(data_path, j)
#         data = dict(json.load(open(path, "r")))   
#         temp.append(data)

# dfNew = pd.DataFrame.from_dict(temp, orient= "columns")   
# # 去除空值
# filter = dfNew["text"] != ""
# df = dfNew[filter]
# X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'keywords', 
#                                                                             train_size=0.8, valid_size=0.1, test_size=0.1)
                        
# X_train.to_json (r'/home/alissa77/practice/WWW2021_dEEFEND/code/preprocess/gossipcop/train.json',  indent = 4, orient="records")
# X_valid.to_json (r'/home/alissa77/practice/WWW2021_dEEFEND/code/preprocess/gossipcop/val.json',  indent = 4, orient="records")
# X_test.to_json (r'/home/alissa77/practice/WWW2021_dEEFEND/code/preprocess/gossipcop/test.json',  indent = 4, orient="records") 
                    
# for i in ["fake", "real"] :
#     data_path = os.path.join(DATA_PATH ,"{}".format(i)) 
#     for j in os.listdir(data_path):
#         path = os.path.join(data_path, j)
#         data = dict(json.load(open(path, "r")))   
#         newpd = pd.DataFrame.from_dict(data, orient= "index")   
# print(newpd)
#         # labels_arr = get_labels_arr(data)
#         # print(labels_arr)
        
        
# train, val, test, train_label, val_label, test_label = train_valid_test_split(data["text"], data["label"], train_size=0.6, valid_size=0.2, test_size=0.2)
# print(train.shape)
# print(train_c.shape)
# print(train_label.shape)

#         # for t in ["train", "val", "test"]:
#         #     with open(os.path.join(output_dir, '{}.json'.format(t)), 'w') as f:
#         #         json.dump(data, f, indent=4, ensure_ascii=False)