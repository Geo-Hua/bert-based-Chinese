import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pylab import mpl
import os
import pandas as pd
import ast
# from central import calcuate as cs
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')

import  math
# emotion label
emotions = ['positive', 'negative']

# calculate entropy
def calculate_entropy(proportions):
    proportions = np.array(proportions)
    proportions = proportions[proportions > 0]  # 去除为0的情绪
    entropy = -np.sum(proportions * np.log2(proportions))
    return entropy


def update_matrix_for_emotion(comment,matrix):
    positive_value = comment[0]['positive']
    negative_value = comment[0]['negative']
    # print(comment)
    emotion_values = np.array([comment[0][emotion] for emotion in emotions])
    idx = np.argmax(emotion_values)

    if positive_value - negative_value>0.05:
         matrix[0, 1] += 1

    elif negative_value-positive_value>0.05:
            # matrix[0, 1] += negative_value
         matrix[1, 0] += 1

    else:
        matrix[0, 0] += 1
        matrix[1, 1] += 1
only=[]
double=[]
half=[]

# Sentiment judgement function, update matrix
def count(comment,matrix):
    positive_value = comment[0]['positive']
    negative_value = comment[0]['negative']
    # print(comment)
    emotion_values = np.array([comment[0][emotion] for emotion in emotions])
    idx = np.argmax(emotion_values)

    if positive_value -negative_value>0.05:
        if negative_value>0.3:
            half.append(1)
        else:
            only.append(1)

    elif negative_value-positive_value>0.05:

        if positive_value > 0.3:
            half.append(1)
        else:
            only.append(1)

    else:
            double.append(1)

# Handling of all comments
def draw_graph(emotion_data,index,graph_dict):
    # Initialising the emotional correlation matrix
    matrix = np.zeros((2, 2))
    for comment in zip(emotion_data):
        update_matrix_for_emotion(comment,matrix)

    graph_dict[index] = matrix



def main(file,part,df1,output_folder):
    folder_path = f'{file}/result/'
    # Determine if a folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    num=len(os.listdir(folder_path))
    # sample_data=[]
    graphs= {}
    # Iterate through all files in a folder
    for filename in os.listdir(folder_path):
        # sample_data = []
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            data.head()
            senti_emotions=data.copy()
            senti_emotion=senti_emotions['pos-neg']
            emotion_data=[]
            entropy_data=[]
            index = senti_emotions['Grid_ID']
            for e in senti_emotion:
                data_dict = ast.literal_eval(e)
                # total = sum(data_dict.values())
                if data_dict:
                    emotion_data.append(data_dict)
            draw_graph(emotion_data,index[0],graphs)
    print(f'单一情绪的长度为{len(only)}')
    print(f'主导附属情绪的长度为{len(half)}')
    print(f'复合情绪的长度为{len(double)}')

    def read_graphs(df,graphs):
        num= df.iloc[-1]['Grid_ID']
        grid_id_range = range(int(num)+1)# 0 to num
        # Initialise a list to store the complete matrix data
        complete_graphs = []
        # Iterate through the Grid_ID range
        for grid_id in grid_id_range:
            if grid_id in graphs:
                complete_graphs.append(graphs[grid_id])
            else:
                complete_graphs.append(np.zeros((2,2)))
        return complete_graphs


    complete_graphs=read_graphs(df1,graphs)
    print(len(complete_graphs))

    # Storing a list to a file
    # Storing a list to a JSON file
    with open(f'{file}/graphs.pkl', 'wb') as f:
        pickle.dump(complete_graphs, f)
    print('program finished')

# Specify folder path
file='../result/bert/wh/128/pos-neg'
output_folder = f"{file}/graph_pos_neg"
# Determine if a folder exists
def ifexist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
ifexist(output_folder)



df_cu = pd.read_csv(f'{file}/grid_lat_lon.csv')
if __name__=="__main__":
    main(file,'cu',df_cu,output_folder)
# c(graphs,num)

