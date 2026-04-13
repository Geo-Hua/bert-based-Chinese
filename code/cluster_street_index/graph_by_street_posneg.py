import numpy as np
import os
import pandas as pd
import ast
import math
import matplotlib.pyplot as plt
import networkx as nx
import pickle



# 情绪标签
emotions = ['positive', 'negative']

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
def update_matrix_for_emotion(comment,matrix):
    positive_value = comment['positive']
    negative_value = comment['negative']
    # print(comment)
    emotion_values = np.array([comment[emotion] for emotion in emotions])
    idx = np.argmax(emotion_values)
        # 当 m > n 时，第1行第2列加上m的值，第2行第1列加上n的值
    if positive_value - negative_value>0.05:
         matrix[0, 1] += 1
            # matrix[1, 0] += negative_value
        # 当 m < n 时，第1行第2列加上n的值，第2行第1列加上m的值
    elif negative_value-positive_value>0.05:
            # matrix[0, 1] += negative_value
         matrix[1, 0] += 1
        # 当 m == n 时，第1行第1列加上m的值，第2行第2列加上n的值
    else:
        matrix[0, 0] += 1
        matrix[1, 1] += 1
only=[]
double=[]
half=[]
# 情绪判断函数，更新矩阵
def count(comment):
    positive_value = comment['positive']
    negative_value = comment['negative']
    # print(comment)
    emotion_values = np.array([comment[emotion] for emotion in emotions])
    idx = np.argmax(emotion_values)
        # 当 m > n 时，第1行第2列加上m的值，第2行第1列加上n的值
    if positive_value -negative_value>0.05:
        if negative_value>0.3:
            half.append(1)
        else:
            only.append(1)
            # matrix[1, 0] += negative_value
        # 当 m < n 时，第1行第2列加上n的值，第2行第1列加上m的值
    elif negative_value-positive_value>0.05:
            # matrix[0, 1] += negative_value
        if positive_value > 0.3:
            half.append(1)
        else:
            only.append(1)
        # 当 m == n 时，第1行第1列加上m的值，第2行第2列加上n的值
    else:
            double.append(1)
def draw_graph(emotion_data, entropy_data, index, graph_dict, output_folder,matrix_emotion):
    matrix = np.zeros((2, 2))
    only = []
    double = []
    half = []
    for comment, entropy in zip(emotion_data, entropy_data):
        update_matrix_for_emotion(comment, matrix)
        count(comment)
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

    for i in range(2):
        for j in range(2):
            if i != j and matrix[i, j] > 0:
                G.add_edge(emotions[i], emotions[j], weight=matrix[i, j])

    pos = nx.spring_layout(G, seed=42, k=10, iterations=50)
    sizes = [G.nodes[node]['size'] for node in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, arrowstyle='->', width=1, alpha=0.5, edge_color='black',
                           arrowsize=20, connectionstyle='arc3,rad=-0.3')

    edge_labels = {(emotions[i], emotions[j]): f'{int(matrix[i, j])}' for i in range(2) for j in range(2) if
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
            senti_emotions = data['pos-neg']
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

            street_name = os.path.splitext(filename)[0]  # 用文件名（去掉 .csv）作为key
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
            {'index': all_index,
             'matrices': complete_graphs
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

g('../result/bert/wh/Street_index/pos-neg/L3/before')