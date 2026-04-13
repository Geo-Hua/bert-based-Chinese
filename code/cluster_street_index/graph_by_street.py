import numpy as np
import os
import pandas as pd
import ast
import math
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import re


# 情绪标签
emotions = ['angry', 'disgust', 'fear', 'like', 'sad', 'surprise', 'happy']

# 清空文件夹的函数
def clear_output_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.png'):
                os.remove(os.path.join(folder_path, filename))
    else:
        os.makedirs(folder_path)

# 计算熵值
def calculate_entropy(proportions):
    proportions = np.array(proportions)
    proportions = proportions[proportions > 0]  # 去除为0的情绪
    entropy = -np.sum(proportions * np.log2(proportions))
    return entropy

# 情绪判断函数，更新矩阵
def update_matrix_for_emotion(comment, entropy, matrix):
    threshold_single = 0.5
    threshold_main = 0.3
    entropy_threshold = 1.5
    emotion_values = np.array([comment[emotion] for emotion in emotions])
    sorted_probs = sorted(emotion_values, reverse=True)
    # max_prob = max(emotion_values)
    max_prob = sorted_probs[0]
    p2 = sorted_probs[1]

    if max_prob > 0.8 and entropy < 0.5:
        idx = np.argmax(emotion_values)
        matrix[idx, idx] += 1

    elif (max_prob > 0.5 and p2>0.2 and entropy<1) or (max_prob+p2>0.7 and entropy<1.25):
        dominant_idx = np.where(emotion_values == max_prob)[0]
        subordinates = np.where(emotion_values != max_prob)[0]
        for i in dominant_idx:
            for sub_idx in subordinates:
                if emotion_values[sub_idx] > 0.1:
                    matrix[i, sub_idx] += 1

    elif np.sum(emotion_values > 0.15) > 2:
        subordinates = np.where(emotion_values > 0.15)[0]
        matrix[np.ix_(subordinates, subordinates)] += 1
        # print(matrix)
        # for i in range(len(subordinates)):
        #     for j in range(i + 1, len(subordinates)):
        #         matrix[i, j] += 1
        #         matrix[j, i] += 1

# 情绪判断函数，更新矩阵
def count(comment, H,only,double,half):
    emotion_values = np.array([comment[emotion] for emotion in emotions])
    sorted_probs = sorted(emotion_values, reverse=True)
    # max_prob = max(emotion_values)
    p1 = sorted_probs[0]
    p2 = sorted_probs[1]
    # sub_prob=max
    if p1 > 0.8 and H < 0.5:
        only.append(1)
    elif (p1 > 0.5 and p2 > 0.2 and H < 1) or (p1 + p2 > 0.7 and H < 1.25):
        double.append(1)
    else:
        half.append(1)
def draw_graph(emotion_data, entropy_data, index, graph_dict, output_folder,matrix_emotion):
    matrix = np.zeros((7, 7))
    only = []
    double = []
    half = []
    for comment, entropy in zip(emotion_data, entropy_data):
        update_matrix_for_emotion(comment, entropy, matrix)
        count(comment, entropy,only,double,half)
    only_num = len(only)
    double_num = len(double)
    half_num = len(half)
    cate = 0
    if only_num > double_num:
        if only_num <= half_num:
            cate = 2
        else:
            cate = 0
    elif only_num <= double_num:
        if double_num <= half_num:
            cate = 2
        else:
            cate = 1
    matrix_emotion[index]=cate
    graph_dict[index] = matrix
    fig, ax = plt.subplots(figsize=(18, 18))
    G = nx.DiGraph()

    for i, emotion in enumerate(emotions):
        G.add_node(emotion, size=matrix[i, i] * 50)

    for i in range(7):
        for j in range(7):
            if i != j and matrix[i, j] > 0:
                G.add_edge(emotions[i], emotions[j], weight=matrix[i, j])

    pos = nx.spring_layout(G, seed=42, k=10, iterations=50)
    sizes = [G.nodes[node]['size'] for node in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, arrowstyle='->', width=1, alpha=0.5, edge_color='black',
                           arrowsize=20, connectionstyle='arc3,rad=-0.3')

    edge_labels = {(emotions[i], emotions[j]): f'{int(matrix[i, j])}' for i in range(7) for j in range(7) if
                   matrix[i, j] > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=15, label_pos=0.4)

    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"graph_{index}.png"), format='png')
    plt.close('all')

def main(file, output_folder):
    folder_path = f'{file}/output_weibo_by_index/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    graphs = {}
    matrix_emotion = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            senti_emotions = data['sentiment']
            emotion_data = []
            entropy_data = []

            for e in senti_emotions:
                data_dict = ast.literal_eval(e)
                total = sum(data_dict.values())
                if total:
                    emotion_ratios = {k: v for k, v in data_dict.items()}
                    emotion_entropy = -sum(p * math.log(p) for p in emotion_ratios.values() if p > 0)
                    emotion_data.append(emotion_ratios)
                    entropy_data.append(emotion_entropy)

            # street_name = os.path.splitext(filename)[0]  # 用文件名（去掉 .csv）作为key
            street_name_raw = os.path.splitext(filename)[0]
            match = re.search(r'\d+', street_name_raw)
            street_name = match.group() if match else street_name_raw
            draw_graph(emotion_data, entropy_data, street_name, graphs, output_folder, matrix_emotion)

    # 将字典转换为 DataFrame，行索引是街道名
    df = pd.DataFrame.from_dict(matrix_emotion, orient='index', columns=['category'])
    df.index.name = 'index'
    df.reset_index(inplace=True)

    csv_file = '../cluster_index.csv'
    df.to_csv(csv_file, index=False)

    # 按照文件中街道顺序生成完整图谱列表
    complete_graphs = []
    all_index = df['index'].tolist()

    for s_index in all_index:
        if s_index in graphs:
            complete_graphs.append(graphs[s_index])
        else:
            complete_graphs.append(np.zeros((7, 7)))

    with open(f'{file}/graphs.pkl', 'wb') as f:
        pickle.dump(
            {   'index':all_index,
                 'matrices':complete_graphs
             }, f)

    print('Program finished')


def g(file):
    # file = 'result/256'
    output_folder_xi = f"{file}/graph"

    def ifexist(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    ifexist(output_folder_xi)
    # clear_output_folder(output_folder_xi)
    #df = pd.read_csv(f'{file}/grid_lat_lon.csv')
    main(file, output_folder_xi)

g('../result/bert/wh/Street_index/pos-neg/L4/before')
# g('../result/bert/sh/Street_index/L4')