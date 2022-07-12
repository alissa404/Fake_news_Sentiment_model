import numpy as np
import os
import json

# only modify results_folder name
results_folder = "/home/alissa77/WWW2021/code/Taiwan_preprocess/train"
results = "EmotionEnhancedBiGRU_test.json"
datasets = "taiwan"
sum = 0
sum2 = 0
fakescore = 0
fakescore2 = 0
realscore = 0
realscore2 = 0

key =7
for i in range(key):
    results_file = os.path.join(results_folder,"{}".format(i))
    results_file = os.path.join(results_file, datasets,results)
    with open(results_file, 'r') as f:
        acc = json.load(f)
        sum += acc['accuracy']
        fakescore += acc['classification_report']['fake']['f1-score']
        realscore += acc['classification_report']['real']['f1-score']
print("Acc :", sum/key)
print("fake :", fakescore/key)
print("real :", realscore/key)
print()
print()