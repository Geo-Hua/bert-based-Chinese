import numpy as np
import pandas as pd
import ast
import math


# label
emotions = ['angry', 'disgust', 'fear', 'like', 'sad', 'surprise', 'happy']
# emotions = ['positive', 'negative']

# calculate entropy
def calculate_entropy(proportions):
    proportions = np.array(proportions)
    proportions = proportions[proportions > 0]  # 去除为0的情绪
    entropy = -np.sum(proportions * np.log2(proportions))
    return entropy
only=[]
double=[]
half=[]

# Sentiment judgement function
def count_7(comment, H):
    cate=0

    emotion_values = np.array([comment[emotion] for emotion in emotions])
    sorted_probs=sorted(emotion_values,reverse=True)
    # max_prob = max(emotion_values)
    p1=sorted_probs[0]
    p2=sorted_probs[1]
    emotion = [k for k, v in comment.items() if v == p1][0]
    if p1 > 0.8 and H<0.6:
        only.append(1)
        if emotion=='angry':
            cate=11
        elif emotion=='disgust':
            cate=12
        elif emotion == 'fear':
            cate = 13
        elif emotion=='like':
            cate=14
        elif emotion=='sad':
            cate=15
        elif emotion=='surprise':
            cate=16
        elif emotion=='happy':
            cate=17
    elif (p1>0.5 and p2>0.2 and H<1) or (p1 + p2 > 0.7 and H < 1.25):
        double.append(1)
        sec_emotion=[k for k, v in comment.items() if v == p2][0]
        if emotion=='angry':
            if sec_emotion=='disgust':
                cate=21
            elif sec_emotion=='fear':
                cate=22
            elif sec_emotion=='like':
                cate=23
            elif sec_emotion=='sad':
                cate=24
            elif sec_emotion=='surprise':
                cate=25
            else:
                cate=26
        elif emotion=='disgust':
            if sec_emotion=='angry':
                cate=31
            elif sec_emotion=='fear':
                cate=32
            elif sec_emotion=='like':
                cate=33
            elif sec_emotion=='sad':
                cate=34
            elif sec_emotion=='surprise':
                cate=35
            else:
                cate=36
        elif emotion == 'fear':
            if sec_emotion == 'angry':
                cate = 41
            elif sec_emotion == 'disgust':
                cate = 42
            elif sec_emotion == 'like':
                cate = 43
            elif sec_emotion == 'sad':
                cate = 44
            elif sec_emotion == 'surprise':
                cate = 45
            else:
                cate = 46
        elif emotion=='like':
            if sec_emotion == 'angry':
                cate = 51
            elif sec_emotion == 'disgust':
                cate = 52
            elif sec_emotion == 'fear':
                cate = 53
            elif sec_emotion == 'sad':
                cate = 54
            elif sec_emotion == 'surprise':
                cate = 55
            else:
                cate = 56
        elif emotion=='sad':
            if sec_emotion == 'angry':
                cate = 61
            elif sec_emotion == 'disgust':
                cate = 62
            elif sec_emotion == 'fear':
                cate = 63
            elif sec_emotion == 'like':
                cate = 64
            elif sec_emotion == 'surprise':
                cate = 65
            else:
                cate = 66
        elif emotion=='surprise':
            if sec_emotion == 'angry':
                cate = 71
            elif sec_emotion == 'disgust':
                cate = 72
            elif sec_emotion == 'fear':
                cate = 73
            elif sec_emotion == 'like':
                cate = 74
            elif sec_emotion == 'sad':
                cate = 75
            else:
                cate = 76
        elif emotion=='happy':
            if sec_emotion == 'angry':
                cate = 81
            elif sec_emotion == 'disgust':
                cate = 82
            elif sec_emotion == 'fear':
                cate = 83
            elif sec_emotion == 'like':
                cate = 84
            elif sec_emotion == 'sad':
                cate = 85
            else:
                cate = 86
    else:
        half.append(1)
        cate=100
    return  cate
pos=[]
neg=[]
netural=[]
def count_2(comment, entropy):
    positive_value = comment['positive']
    negative_value = comment['negative']
    # print(comment)
    emotion_values = np.array([comment[emotion] for emotion in emotions])
    idx = np.argmax(emotion_values)

    if positive_value - negative_value > 0.05:
            pos.append(1)

    elif negative_value - positive_value > 0.05:
        # matrix[0, 1] += negative_value
        neg.append(1)

    else:
        netural.append(1)
def count(file_path):
    # file_path = 'result/bert/wh/emotion_prediction_wh.csv'
    data = pd.read_csv(file_path)
    #senti_emotions = data['pos-neg']
    senti_emotions = data['sentiment']

    category=[]
    for e in senti_emotions:
        data_dict = ast.literal_eval(e)
        total = sum(data_dict.values())
        if total:
            # emotion_ratios = {k: v / total for k, v in data_dict.items()}
            emotion_ratios = {k: v for k, v in data_dict.items()}
            emotion_entropy = -sum(p * math.log(p) for p in emotion_ratios.values() if p > 0)
            # emotion_data.append(emotion_ratios)
            # entropy_data.append(emotion_entropy)
            cate=count_7(emotion_ratios, emotion_entropy)
            #count_2(emotion_ratios, emotion_entropy)
            category.append(cate)
    print(f'{list(set(category))}')
    data['cate']=category
    data.to_csv('../data/emotion_prediction_wh.csv', index=False)
    print(f'单一情绪的长度为{len(only)}')
    print(f'主导附属情绪的长度为{len(double)}')
    print(f'复合情绪的长度为{len(half)}')
    print(f'积极的长度为{len(pos)}')
    print(f'消极的长度为{len(neg)}')
    print(f'中性的长度为{len(netural)}')

if __name__=="__main__":
    count('../data/emotion_prediction_wh.csv')