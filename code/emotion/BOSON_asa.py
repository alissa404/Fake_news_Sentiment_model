# In DEAN the highest accuracy!

# with input_of_emotions_test
# NOT mix bosonNLP and sentiment score in one output
import joblib
import pandas as pd
import numpy as np
import jieba
# ============================== Category ==============================
baidu_emotions = ['angry', 'disgusting', 'fearful',
                  'happy', 'sad', 'neutral', 'pessimistic', 'optimistic']
baidu_emotions.sort()
baidu_emotions_2_index = dict(
    zip(baidu_emotions, [i for i in range(len(baidu_emotions))]))

def baidu_arr(emotions_dict):
    arr = np.zeros(len(baidu_emotions))
    if emotions_dict is None:
        return arr

    for k, v in emotions_dict.items():
        # like -> happy
        if k == 'like':
            arr[baidu_emotions_2_index['happy']] += v
        else:
            arr[baidu_emotions_2_index[k]] += v

    return arr

##################### load negation words #####################
negation_words = []
with open('../../resources/Chinese/others/negative/negationWords.txt', 'r') as src:
    lines = src.readlines()
    for line in lines:
        negation_words.append(line.strip())

print('\nThe num of negation words: ', len(negation_words))
prices = [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,
        0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8]
neg_words_dict = dict(zip(negation_words, prices))
negative = neg_words_dict

# ###################### load degree words #####################
how_words_dict = dict()
with open('../../resources/Chinese/HowNet/intensifierWords.txt', 'r') as src:
    lines = src.readlines()
    for line in lines:
        how_word = line.strip().split()
        how_words_dict[' '.join(how_word[:-1])] = float(how_word[-1])

print('The num of degree words: ', len(how_words_dict),'. eg: ', list(how_words_dict.items())[0])
print()
degree = how_words_dict

###################### negation value and degree value #####################

# 程度副詞 #md
degree2 = {'最':1.9, '最为':1.9, '极':1.9, '极为':1.9, '极其':1.9, '极度':1.9, '极端':1.9, '剧烈':1.9,\
           '顶':1.9, '过':1.9, '过于':1.9, '过分':1.9, '分外':1.9, '万分':1.9,"全":1.9,'千万':1.9,'绝对':1.9,\
           '更':1.7, '更加':1.7, '更为':1.7, '更其':1.7, '越':1.7, '越发':1.7, '备加':1.7, '愈加':1.7, '愈':1.7,\
           '愈发':1.7, '愈为':1.7, '愈益':1.7, '越加':1.7, '益发':1.7, '还':1.5, '很':1.5, '太':1.7,'都':1.7,\
           '挺':1.7, '怪':1.7, '老是':1.7, '非常':1.7, '特别':1.7, '十分':1.7, '好':1.7, '好不':1.7,\
           '甚':1.7, '甚为':1.7, '颇':1.7, '颇为':1.7, '满':1.7, '蛮':1.7, '够':1.7, '多':1.7,\
           '多么':1.7, '特':1.7, '大':1.7, '大为':1.7, '何等':1.7, '何其':1.7, '尤其':1.7, '无比':1.7, '尤为':0.3,\
           '较':1.5, '比较':1.5, '较比':1.5, '较为':1.5, '还':1.5, '不大':0.5, '不太':1.5, '不很':1.5, '不甚':1.5,\
           '早已':1.7,'吗':1.5,'大多':0.5,'超级':1.7, '一点':1.5, '至少':1.5,'绝':1.5,\
           '稍':0.5, '稍稍':0.5, '稍为':0.5, '稍微':0.5, '稍许':0.5, '亲自':0.5, '略':0.5, '略为':0.5,\
           '些微':0.5, '多少':0.5, '有点':0.7, '有点儿':0.5, '有些':0.5, '多为':0.7, '相当':0.7, '至为':0.7,}

# 否定副词 #mn
negative2 = {'白':0.8, '白白':0.8, '甭':0.8, '别':0.8, '不':0.8, '不必':0.8, '不曾':0.8, '不太':0.8,\
            '不用':0.8, '非':0.8, '干':0.8, '何必':0.8, '何曾':0.8, '何尝':0.8, '何须':0.8,\
            '空':0.8, '没':0.8, '没有':0.8, '莫':0.8, '徒':0.8, '徒然':0.8, '忹':0.8,'不要':0.8, '不堪': 0.8,\
            '未':0.8, '未曾':0.8, '未尝':0.8, '无须':0.8, '无须乎':0.8, '无需':0.8, '毋须':0.8,\
            '毋庸':0.8, '无庸':0.8, '勿':0.8, '瞎':0.8, '休':0.8, '虚':0.8, '假':0.8, '也不':0.8}

degree.update(degree2)
negative.update(negative2)

######################### negation value and degree value ##############################

def get_not_and_how_value(cut_words, i, windows):
    not_cnt2 = 0
    how_v2 = 0
    left = 0 if (i - windows) < 0 else (i - windows)
    window_text = ' '.join(cut_words[left:i])
    for w in negative.keys():
        if w in window_text:
            not_cnt2 += negative[w]
            #not_cnt2/2
                               
    for w in degree.keys():
        if w in window_text:
            how_v2 += degree[w]
            #how_v2/2

    return not_cnt2.real, (-1) ** how_v2.real



def times(cut_words, i, windows):
    not_cnt = 0
    how_v = 0
    left = 0 if (i - windows) < 0 else (i - windows)
    window_text = ' '.join(cut_words[left:i])
    for w in negative:
        if w in window_text:
            not_cnt += 1
    for w in degree:
        if w in window_text:
            how_v += 1
    return not_cnt, how_v        

################################# Boson_NLP ####################################

boson_words_dict = {}
with open('../../resources/Chinese/BosonNLP/BosonNLP_sentiment_score.txt', 'r') as src:
#with open('/home/alissa77/WWW2021/resources/Chinese/high_aro_boson.txt', 'r') as src:
    lines = src.readlines()
    for line in lines:
        boson_word = line.strip().split()
        if len(boson_word) != 2:
            continue
        else:
            boson_words_dict[boson_word[0]] = float(boson_word[1])*(-1)

print('[BosonNLP]\t There are {} words'.format(len(boson_words_dict)))
arousal = boson_words_dict 

################################# Sentiment ####################################
def sentiment_score(cut_words, deg_words, neg_words, aro_words, max_compare):
    arr = np.zeros((37, max_compare)) # max_compare 每筆資料對於 每一個pattern要比的次數
    index = np.zeros(37).astype(int)
    start_index = 0 #第一個字
    end_index = 3 #第三個字
    
    while end_index <= len(cut_words):
        words = cut_words[start_index:end_index]
        # 3 words pattern deg, neg, aro
        if deg_words.get(words[0]) and neg_words.get(words[1]) and aro_words.get(words[2]):
            start_index = end_index
            end_index += 3
            if index[0] < max_compare:
                arr[0][index[0]] = deg_words[words[0]] * neg_words[words[1]] * aro_words[words[2]]
                index[0] += 1
                print(words)
            if index[1] < max_compare:
                arr[1][index[1]] = deg_words[words[0]] * neg_words[words[1]] + aro_words[words[2]]
                index[1] += 1
                print(words)

            if index[2] < max_compare:
                arr[2][index[2]] = (deg_words[words[0]] + neg_words[words[1]]) * aro_words[words[2]]
                index[2] += 1
                print(words)
                
            if index[3] < max_compare:
                arr[3][index[3]] = deg_words[words[0]] + neg_words[words[1]] + aro_words[words[2]]
                index[3] += 1
            if index[4] < max_compare:
                arr[4][index[4]] = deg_words[words[0]] + neg_words[words[1]] * aro_words[words[2]]
                index[4] += 1                
            continue
        # 3 words pattern aro, deg, neg
        if aro_words.get(words[0]) and deg_words.get(words[1]) and neg_words.get(words[2]):
            start_index = end_index
            end_index += 3
            if index[5] < max_compare:
                arr[5][index[5]] = aro_words[words[0]] * deg_words[words[1]] + neg_words[words[2]] 
                index[5] += 1
            if index[6] < max_compare:
                arr[6][index[6]] = aro_words[words[0]] + deg_words[words[1]] + neg_words[words[2]] 
                index[6] += 1
                print(words)
                
            if index[7] < max_compare:
                arr[7][index[7]] = aro_words[words[0]] * (deg_words[words[1]] + neg_words[words[2]]) 
                index[7] += 1
                print(words)

            if index[8] < max_compare:
                arr[8][index[8]] = aro_words[words[0]] + deg_words[words[1]] * neg_words[words[2]] 
                index[8] += 1
            if index[9] < max_compare:
                arr[9][index[9]] = aro_words[words[0]] * deg_words[words[1]] * neg_words[words[2]] 
                index[9] += 1
            continue


        # 3 words pattern aro, deg, deg
        if aro_words.get(words[0]) and deg_words.get(words[1]) and deg_words.get(words[2]):
            start_index = end_index
            end_index += 3
            if index[18] < max_compare:
                arr[18][index[18]] = aro_words[words[0]] * (deg_words[words[1]] + deg_words[words[2]])
                index[18] += 1
            if index[19] < max_compare:
                arr[19][index[19]] = aro_words[words[0]] + deg_words[words[1]] + deg_words[words[2]]
                index[19] += 1
            if index[20] < max_compare:
                arr[20][index[20]] = aro_words[words[0]] * deg_words[words[1]] * deg_words[words[2]]
                index[20] += 1
            continue

            
            
        words = cut_words[start_index:end_index-1]
        
        # 2 words pattern neg, aro
        if neg_words.get(words[0]) and aro_words.get(words[1]):
            start_index = end_index - 1
            end_index += 2
            if index[29] < max_compare:
                arr[29][index[29]] = neg_words[words[0]] * aro_words[words[1]]
                index[29] += 1
                print(words)

            if index[30] < max_compare:
                arr[30][index[30]] = neg_words[words[0]] + aro_words[words[1]]
                index[30] += 1
            continue
        # 2 words pattern, deg, aro
        if deg_words.get(words[0]) and aro_words.get(words[1]):
            start_index = end_index - 1
            end_index += 2
            if index[31] < max_compare:
                arr[31][index[31]] = deg_words[words[0]] * aro_words[words[1]]
                index[31] += 1
            if index[32] < max_compare:
                arr[32][index[32]] = deg_words[words[0]] + aro_words[words[1]]
                index[32] += 1
            continue
        # 2 words pattern, aro, deg
        if aro_words.get(words[0]) and deg_words.get(words[1]):
            start_index = end_index - 1
            end_index += 2
            if index[33] < max_compare:
                arr[33][index[33]] = aro_words[words[0]] * deg_words[words[1]]
                index[33] += 1
            if index[34] < max_compare:
                arr[34][index[34]] = aro_words[words[0]] + deg_words[words[1]]
                index[34] += 1
            continue
        # 2 words pattern aro, neg
        if aro_words.get(words[0]) and neg_words.get(words[1]):
            start_index = end_index - 1
            end_index += 2
            if index[35] < max_compare:
                arr[35][index[35]] = aro_words[words[0]] * neg_words[words[1]]
                index[35] += 1
            if index[36] < max_compare:
                arr[36][index[36]] = aro_words[words[0]] + neg_words[words[1]]
                index[36] += 1    
            continue
        start_index += 1
        end_index += 1
    return arr.flatten()



def boson_value(cut_words, windows=2):
    value2 = 0
    for i, word in enumerate(cut_words):
        if word in arousal:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            value2 += not_v + how_v + arousal[word]
    return round(value2.real,10)

# ============================== dalianligong ==============================

_, words2array = joblib.load(
    '../../resources/Chinese/大连理工大学情感词汇本体库/preprocess/words2array_27351.pkl')
print('[Dalianligong]\tThere are {} words'.format(len(words2array)))
print()


def dalianligong_arr(cut_words, windows=2):
    arr = np.zeros(29)

    for i, word in enumerate(cut_words):
        if word in words2array:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            arr = arr + not_v + how_v + words2array[word]

    return arr

# ============================== Sentiment Scores ==============================

# Sentimental Words 2.0
def init_words(file):
    with open(file, 'r', encoding='utf-8') as src:
        words = src.readlines()
        words = [l.strip() for l in words]

    return list(set(words))

pos_words = init_words('../../resources/Chinese/HowNet/正面情感词语（中文）.txt')
pos_words += init_words('../../resources/Chinese/HowNet/正面评价词语（中文）.txt')
neg_words = init_words('../../resources/Chinese/HowNet/负面情感词语（中文）.txt')
neg_words += init_words('../../resources/Chinese/HowNet/负面评价词语（中文）.txt')
deg_words = init_words('../../resources/Chinese/HowNet/负面评价词语（中文）.txt')

pos_words = set(pos_words)
neg_words = set(neg_words)
deg_words = set(deg_words)
print('[HowNet]\tThere are {} positive words and {} negative words'.format(len(pos_words), len(neg_words), len(deg_words)))
print()


def sentiment_words_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0, 0]

    # positive and negative words
    sentiment = []
    for words in [pos_words, neg_words]:
        c = 0
        for word in words:
            if word in cut_words:
                # print(word)
                c += 1
        sentiment.append(c)
    sentiment = [c / len(cut_words) for c in sentiment]

    # degree words
    degree = 0
    for word in how_words_dict:
        if word in cut_words:
            # print(word)
            degree += how_words_dict[word]

    # negation words
    negation = 0
    for word in negation_words:
        negation += cut_words.count(word)
    negation /= len(cut_words)
    sentiment += [degree, negation]

    return sentiment


# Personal Pronoun 你我他
first_pronoun = init_words(
    '../../resources/Chinese/others/pronoun/1-personal-pronoun.txt')
second_pronoun = init_words(
    '../../resources/Chinese/others/pronoun/2-personal-pronoun.txt')
third_pronoun = init_words(
    '../../resources/Chinese/others/pronoun/3-personal-pronoun.txt')
pronoun_words = [first_pronoun, second_pronoun, third_pronoun]


def pronoun_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]
    pronoun = []
    for words in pronoun_words:
        c = 0
        for word in words:
            c += cut_words.count(word)
        pronoun.append(c)

    return [c / len(cut_words) for c in pronoun]

# # ============================== Auxilary Features ==============================

# Emotion
emoticon_df = pd.read_csv(
    '../../resources/Chinese/others/emoticon/emoticon.csv')
emoticons = emoticon_df['emoticon'].tolist()
emoticon_types = list(set(emoticon_df['label'].tolist()))
emoticon_types.sort()
emoticon2label = dict(
    zip(emoticon_df['emoticon'].tolist(), emoticon_df['label'].tolist()))
emoticon2index = dict(
    zip(emoticon_types, [i for i in range(len(emoticon_types))]))

print('[Emoticon]\tThere are {} emoticons, including {} categories'.format(
    len(emoticons), len(emoticon_types)))


def emoticon_arr(text, cut_words):
    arr = np.zeros(len(emoticon_types))
    if len(cut_words) == 0:
        return arr
    for i, emoticon in enumerate(emoticons):
        if emoticon in text:
            arr[emoticon2index[emoticon2label[emoticon]]] += text.count(emoticon)

    return arr / len(cut_words)

# Punctuation
def Punctuation(text):
    excl = (text.count('!') + text.count('！')) / len(text)
    ques = (text.count('?') + text.count('？')) / len(text)
    comma = (text.count(',') + text.count('，')) / len(text)
    dot = (text.count('.') + text.count('。')) / len(text)
    ellip = (text.count('..') + text.count('。。')) / len(text)
    #quesexcl = (text.count('?!') + text.count('？！')) / len(text)

    return excl, ques, comma, dot, ellip


# Sentimental Words
def init_words(file):
    with open(file, 'r', encoding='utf-8') as src:
        words = src.readlines()
        words = [l.strip() for l in words]
    # print('File: {}, Words_sz = {}'.format(file.split('/')[-1], len(words)))
    return list(set(words))


pos_words = init_words('../../resources/Chinese/HowNet/正面情感词语（中文）.txt')
pos_words += init_words('../../resources/Chinese/HowNet/正面评价词语（中文）.txt')
neg_words = init_words('../../resources/Chinese/HowNet/负面情感词语（中文）.txt')
neg_words += init_words('../../resources/Chinese/HowNet/负面评价词语（中文）.txt')

pos_words = set(pos_words)
neg_words = set(neg_words)
print('[HowNet]\tThere are {} positive words and {} negative words'.format(
    len(pos_words), len(neg_words)))


def sentiment_words_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0, 0]

    # positive and negative words
    sentiment = []
    for words in [pos_words, neg_words]:
        c = 0
        for word in words:
            if word in cut_words:
                # print(word)
                c += 1
        sentiment.append(c)
    sentiment = [c / len(cut_words) for c in sentiment]

    # degree words
    degree = 0
    for word in how_words_dict:
        if word in cut_words:
            degree += how_words_dict[word]

    # negation words
    negation = 0
    for word in negation_words:
        negation += cut_words.count(word)
    negation /= len(cut_words)
    sentiment += [degree, negation]

    return sentiment


# Personal Pronoun
first_pronoun = init_words(
    '../../resources/Chinese/others/pronoun/1-personal-pronoun.txt')
second_pronoun = init_words(
    '../../resources/Chinese/others/pronoun/2-personal-pronoun.txt')
third_pronoun = init_words(
    '../../resources/Chinese/others/pronoun/3-personal-pronoun.txt')
pronoun_words = [first_pronoun, second_pronoun, third_pronoun]


def pronoun_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]
    
    pronoun = []
    for words in pronoun_words:
        c = 0
        for word in words:
            c += cut_words.count(word)
        pronoun.append(c)

    return [c / len(cut_words) for c in pronoun]


# Auxilary Features
def auxilary_features(text, cut_words):
    arr = np.zeros(17)
    arr[:5] = emoticon_arr(text, cut_words)
    arr[5:10] = Punctuation(text)
    arr[10:14] = sentiment_words_count(cut_words)
    arr[14:17] = pronoun_count(cut_words)

    return arr

# # ============================== Main ==============================
def cut_words_from_text(text):
    return list(jieba.cut(text))

def extract_publisher_emotion(content, content_words, emotions_dict):
    text, cut_words = content, content_words
    arr = np.zeros(100)
    
    arr[:8] = baidu_arr(emotions_dict)
    arr[8:37] = dalianligong_arr(cut_words)
    arr[37:38] = boson_value(cut_words)
    arr[38:55] = auxilary_features(text, cut_words)
    arr[55:92] = sentiment_score(cut_words, degree, negative, arousal, 1)

    return arr

def extract_dual_emotion(piece):
    publisher_emotion = extract_publisher_emotion(piece['content'], piece['content_words'], piece['content_emotions'])

    return publisher_emotion

def extract_title_emotion(piece):
    publisher_emotion = extract_publisher_emotion(piece['title'], piece['title_words'], piece['content_emotions'])

    return publisher_emotion