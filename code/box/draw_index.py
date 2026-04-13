import pickle
import numpy as np
import networkx as nx
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 1. 读取 7×7 矩阵列表
with open('./result/bert/wh/128/graphs.pkl', 'rb') as f:
    matrix_list = pickle.load(f)

# 2. 读取格网边界坐标
grid_df = pd.read_csv('./result/bert/wh/128/grid_lat_lon.csv')  # 包含 Grid_ID, Lat_Min, Lat_Max, Lon_Min, Lon_Max
grid_df['lat'] = (grid_df['Lat_Min'] + grid_df['Lat_Max']) / 2
grid_df['lon'] = (grid_df['Lon_Min'] + grid_df['Lon_Max']) / 2

# 3. 定义图指标计算函数
def compute_graph_metrics(matrix):
    mat = np.array(matrix)
    mat = np.maximum(mat, mat.T)  # 保证对称
    G = nx.from_numpy_array(mat)

    degree = np.mean([v for k, v in G.degree(weight='weight')])
    pagerank = np.mean(list(nx.pagerank(G, weight='weight').values()))
    closeness = np.mean(list(nx.closeness_centrality(G).values()))
    clustering = np.mean(list(nx.clustering(G, weight='weight').values()))

    return degree, pagerank, closeness, clustering

# 4. 遍历每个格网，计算图指标
metrics = []
for idx, matrix in enumerate(matrix_list):
    deg, pr, clo, clu = compute_graph_metrics(matrix)
    row = grid_df.iloc[idx]
    metrics.append({
        'Grid_ID': row['Grid_ID'],
        'Lat_Min':row['Lat_Min'],
        'Lat_Max':row['Lat_Max'],
        'Lon_Min':row['Lon_Min'],
        'Lon_Max':row['Lon_Max'],
        'lat': row['lat'],
        'lon': row['lon'],
        'degree': deg,
        'pagerank': pr,
        'closeness': clo,
        'clustering': clu
    })


def normalize_column(df, col):
    min_val = df[col].min()
    max_val = df[col].max()
    df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val + 1e-9)
    return df
def log_transform_column(df, col):
    df[col + '_log'] = np.log10(df[col] + 1e-10)
    return df

metrics_df = pd.DataFrame(metrics)
# 归一化
for col in ['degree', 'pagerank', 'closeness', 'clustering']:
    metrics_df = normalize_column(metrics_df, col)

# 可选 log 转换（pagerank / clustering）
metrics_df = log_transform_column(metrics_df, 'pagerank')
metrics_df = log_transform_column(metrics_df, 'clustering')


metrics_df.to_csv('./result/bert/wh/128/grid_graph_metrics.csv', index=False)
import folium
import branca.colormap as cm

def scientific_format(val):
    return f"{val:.2e}" if abs(val) < 0.01 else f"{val:.2f}"

def plot_grid_with_values(df, value_col, title, html_file):
    # 色带范围
    colormap = cm.linear.YlGnBu_09.scale(df[value_col].min(), df[value_col].max())

    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=11, tiles='cartodbpositron')
    colormap.caption = title
    colormap.add_to(m)

    for _, row in df.iterrows():
        bounds = [
            [row['Lat_Min'], row['Lon_Min']],
            [row['Lat_Min'], row['Lon_Max']],
            [row['Lat_Max'], row['Lon_Max']],
            [row['Lat_Max'], row['Lon_Min']],
            [row['Lat_Min'], row['Lon_Min']]
        ]
        val = row[value_col]
        label = scientific_format(val)

        folium.Polygon(
            locations=bounds,
            color='black',
            weight=0.5,
            fill=True,
            fill_color=colormap(val),
            fill_opacity=0.7
        ).add_to(m)

        folium.map.Marker(
            [row['lat'], row['lon']],
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size:8px; color:black; text-align:center;">{label}</div>',
            )
        ).add_to(m)

    m.save(html_file)
    print(f"{title} 已保存为 {html_file}")
plot_grid_with_values(metrics_df, 'degree', '平均度数', './result/bert/wh/128/degree_map.html')
plot_grid_with_values(metrics_df, 'pagerank', 'PageRank', './result/bert/wh/128/pagerank_map.html')
plot_grid_with_values(metrics_df, 'closeness', 'Closeness Centrality', './result/bert/wh/128/closeness_map.html')
plot_grid_with_values(metrics_df, 'clustering', '聚类系数', './result/bert/wh/128/clustering_map.html')
