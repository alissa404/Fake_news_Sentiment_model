import numpy as np
import os
import json

# only modify results_folder name
results_folder = "/home/alissa77/WWW2021 copy/code/train/results"
results = "EmotionEnhancedBiGRU_test.json"
sum = 0
sum2 = 0
fakescore = 0
fakescore2 = 0
realscore = 0
realscore2 = 0

def count(datasets, results_dir):
    results_folder = os.path.join ("/home/alissa77/WWW2021 copy/code/train/", results_dir)
    for i in range(10):
        results_file = os.path.join(results_folder,"results{}".format(i))
        results_file = os.path.join(results_file, "./",datasets,"/EmotionEnhancedBiGRU_test.json")
        with open(results_file, 'r') as f:
            acc = json.load(f)
            sum += acc['accuracy']
            fakescore += acc['classification_report']['fake']['f1-score']
            realscore += acc['classification_report']['real']['f1-score']
    return datasets, sum/10, fakescore/10, realscore/10


# Weibo-16
for i in range(10):
    results_file = os.path.join(results_folder,"results{}".format(i))
    results_file = os.path.join(results_file, "./Weibo-16/EmotionEnhancedBiGRU_test.json")
    with open(results_file, 'r') as f:
        acc = json.load(f)
        sum += acc['accuracy']
        fakescore += acc['classification_report']['fake']['f1-score']
        realscore += acc['classification_report']['real']['f1-score']
print("Acc Weibo16:", sum/10)
print("fake Weibo16:", fakescore/10)
print("real Weibo16:", realscore/10)
print()
print()

# Weibo-20
for i in range(10):
    results_file2 = os.path.join(results_folder,"results{}".format(i))
    results_file2 = os.path.join(results_file2, "./Weibo-20/EmotionEnhancedBiGRU_test.json")
    with open(results_file2, 'r') as f2:
        acc2 = json.load(f2)
        sum2 += acc2['accuracy']
        fakescore2 += acc['classification_report']['fake']['f1-score']
        realscore2 += acc['classification_report']['real']['f1-score']
print("final score Weibo20:", sum2/10)
print("fake Weibo20:", fakescore2/10)
print("real Weibo20:", realscore2/10)