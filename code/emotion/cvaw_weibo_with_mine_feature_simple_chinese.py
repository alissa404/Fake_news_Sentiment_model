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
prices = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
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
degree2 = {'最':1.9, '最为':1.9, '极':1.9, '极为':1.9, '极其':1.9, '极度':1.9, '极端':1.9, '剧烈':1.9\
           '顶':1.9, '过':1.9, '过于':1.9, '过分':1.9, '分外':1.9, '万分':1.9,"全":1.9,'千万':1.9,'绝对':1.9,\
           '更':1.7, '更加':1.7, '更为':1.7, '更其':1.7, '越':1.7, '越发':1.7, '备加':1.7, '愈加':1.7, '愈':1.7,\
           '愈发':1.7, '愈为':1.7, '愈益':1.7, '越加':1.7, '益发':1.7, '还':1.5, '很':1.5, '太':1.7,'都':1.7,\
           '挺':1.7, '怪':1.7, '老是':1.7, '非常':1.7, '特别':1.7, '十分':1.7, '好':1.7, '好不':1.7,\
           '甚':1.7, '甚为':1.7, '颇':1.7, '颇为':1.7, '满':1.7, '蛮':1.7, '够':1.7, '多':1.7,\
           '多么':1.7, '特':1.7, '大':1.7, '大为':1.7, '何等':1.7, '何其':1.7, '尤其':1.7, '无比':1.7, '尤为':0.3,\
           '较':1.5, '比较':1.5, '较比':1.5, '较为':1.5, '还':1.5, '不大':0.5, '不太':1.5, '不很':1.5, '不甚':1.5,\
           '早已':1.7,'吗':1.5,'大多':0.5,'超级':1.7, '一点':1.5, '至少':1.5,'绝':1.5,\
           '稍':0.5, '稍稍':0.5, '稍为':0.5, '稍微':0.5, '稍许':0.5, '亲自':0.5, '略':0.5, '略为':0.5,\
           '些微':0.5, '多少':0.5, '有点':0.7, '有点儿':0.5, '有些':0.5, '多为':0.7, '相当':0.7, '至为':0.7}

# 否定副词 #mn
negative2 = {'白':1, '白白':1, '甭':1, '别':1, '不':1, '不必':1, '不曾':1, '不太':1,\
            '不用':1, '非':1, '干':1, '何必':1, '何曾':1, '何尝':1, '何须':1,\
            '空':1, '没':1, '没有':1, '莫':1, '徒':1, '徒然':1, '忹':1,'不要':1, '不堪': 1,\
            '未':1, '未曾':1, '未尝':1, '无须':1, '无须乎':1, '无需':1, '毋须':1,\
            '毋庸':1, '无庸':1, '勿':1, '瞎':1, '休':1, '虚':1, '假':1, '也不':1, '穷':1}


degree.update(degree2)
negative.update(negative2)

######################### negation value and degree value ##############################

def get_not_and_how_value(cut_words, i, windows):
    not_cnt = 0
    how_v = 1
    left = 0 if (i - windows) < 0 else (i - windows)
    window_text = ' '.join(cut_words[left:i])
    for w in negation_words:
        if w in window_text:
            not_cnt += 1
    for w in how_words_dict.keys():
        if w in window_text:
            how_v *= how_words_dict[w]
    return (-1) ** not_cnt, how_v


################################# CVAW ####################################
import csv
def sesntiment_score (cut_words, windows=2):
    def b_edge(val):
        if val<0:
            return -4
        else:
            return 4     
    
    valence = {}
    arousal = {}
    with open('/home/alissa77/WWW2021 copy/code/emotion/cvaw4 simple chinese.csv', newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            valence[row['Word']] = (row['Valence_Mean'])
            arousal[row['Word']] = (row['Arousal_Mean'])

    valence = {x: round(float(valence[x])-5, 2) for x in valence}
    arousal = {x: round(float(arousal[x])-5, 2) for x in arousal}

    lyrics = cut_words
    for i in range(2, len(lyrics)):
        score_a = 0
        score_degree = 0
        score_negative = 0
        len_cvaw = 1 
        lyric = lyrics
        temp_a = 0  # aep

        if  lyrics in list(degree):
            score_degree += degree[lyrics]
        if  lyrics in list(negative):
            score_negative += negative[lyrics]

        if lyric[i] in arousal:
            len_cvaw += 1
            this_a = arousal[lyric[i]]  # Aew

            if lyric[i-2] in negative:
                if lyric[i-1] in negative:
                    print ("N + N + EW:", lyric[i-2:i+1:])
                    param = negative[ lyric[i-2] ] * negative[ lyric[i-1] ]
                    temp_a = param * this_a

                elif lyric[i-1] in degree:
                    print ("N + D + EW:", lyric[i-2:i+1:])
                    param = degree[ lyric[i-1] ] - (1 + negative[ lyric[i-2] ])
                    temp_a = this_a + (b_edge(this_a) - this_a) * param

            elif lyric[i-2] in degree:
                if lyric[i-1] in negative:
                    print ("D + N + EW:", lyric[i-2:i+1:])
                    mn = negative[ lyric[i-1] ]
                    md = degree[ lyric[i-2] ]
                    param_a = mn * this_a
                    temp_a = param_a + (b_edge(this_a) - param_a) * md

                elif lyric[i-1] in degree:
                    print ("D + D + EW:", lyric[i-2:i+1:])
                    md_1 = degree[ lyric[i-1] ]
                    md_2 = degree[ lyric[i-2] ]
                    param_a = (b_edge(this_a) - this_a) * md_1
                    temp_a = this_a + param_a + (1 - (this_a + param_a)) * md_2

            elif lyric[i-1] in negative:
                print ("N + EW:", lyric[i-1:i+1:])
                temp_a = negative[ lyric[i-1] ] * arousal[lyric[i]]

            elif lyric[i-1] in degree:
                print ("D + EW:", lyric[i-1:i+1:])
                temp_a = this_a + (b_edge(this_a) - this_a) * degree[ lyric[i-1] ]
            
            else:
                print ("EW:", lyric[i])   
                temp_a = arousal[lyric[i]]
        
            score_a += temp_a             
        return score_a



##read resource
valence = {}
arousal = {}
with open('/home/alissa77/WWW2021 copy/code/emotion/cvaw4 simple chinese.csv', newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        valence[row['Word']] = (row['Valence_Mean'])
        arousal[row['Word']] = (row['Arousal_Mean'])

valence = {x: round(float(valence[x])-5, 2) for x in valence}
arousal = {x: round(float(arousal[x])-5, 2) for x in arousal}

def boson_value(cut_words, windows=2):
    value = 0

    for i, word in enumerate(cut_words):
        if word in arousal:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            value += not_v * how_v * arousal[word]

    return value


_, words2array = joblib.load(
    '../../resources/Chinese/大连理工大学情感词汇本体库/preprocess/words2array_27351.pkl')
print('[Dalianligong]\tThere are {} words, the dimension is {}'.format(
    len(words2array), words2array['快乐'].shape))


def dalianligong_arr(cut_words, windows=2):
    arr = np.zeros(29)

    for i, word in enumerate(cut_words):
        if word in words2array:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            arr += not_v * how_v * words2array[word]

    return arr


# ============================== Auxilary Features ==============================

# Emoticon
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
            arr[emoticon2index[emoticon2label[emoticon]]
                ] += text.count(emoticon)

    return arr / len(cut_words)


# Punctuation
def symbols_count(text):
    excl = (text.count('!') + text.count('！')) / len(text)
    ques = (text.count('?') + text.count('？')) / len(text)
    comma = (text.count(',') + text.count('，')) / len(text)
    dot = (text.count('.') + text.count('。')) / len(text)
    ellip = (text.count('..') + text.count('。。')) / len(text)

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
                print(word)
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
    print("Hownet : ", sentiment)
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
    arr[5:10] = symbols_count(text)
    arr[10:14] = sentiment_words_count(cut_words)
    arr[14:17] = pronoun_count(cut_words)

    return arr


# ============================== Main ==============================

def cut_words_from_text(text):
    return list(jieba.cut(text))


def extract_publisher_emotion(content, content_words):
    text, cut_words = content, content_words
    arr = np.zeros(56)
    arr[8:37] = dalianligong_arr(cut_words)
    arr[37:38] = boson_value(cut_words)
    arr[38:55] = auxilary_features(text, cut_words)
    arr[55:56] = sesntiment_score(cut_words)

    arr[55:56][np.isnan(arr[55:56])]=0
    
    return arr

def extract_dual_emotion(piece):
    publisher_emotion = extract_publisher_emotion(piece['content'], piece['content_words'])

    return publisher_emotion