import json
import pickle
import numpy as np
import networkx as nx
import pandas as pd
import folium
from folium.plugins import MarkerCluster

stage=''
# 1. 读取 7×7 矩阵列表
with open(f'./result/bert/wh/128/{stage}/graphs.pkl', 'rb') as f:
    matrix_list = pickle.load(f)

# 2. 读取格网边界坐标
grid_df = pd.read_csv(f'./result/bert/wh/128/{stage}/grid_lat_lon.csv')  # 包含 Grid_ID, Lat_Min, Lat_Max, Lon_Min, Lon_Max
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


metrics_df.to_csv(f'./result/bert/wh/128/{stage}/grid_graph_metrics1.csv', index=False)
import folium
import branca.colormap as cm

def scientific_format(val):
    return f"{val:.2e}" if abs(val) < 0.01 else f"{val:.2f}"

def plot_grid_with_values(df, value_col, title, html_file):
    # 色盲友好的色带：OrRd / Viridis / Cividis / Inferno（推荐）
    # 设置统一色带区间（以 degree 为例）
    # colormap_ranges = {
    #     'degree': (0, 160),
    #     'clustering': (0, 1),
    #     'closeness': (0, 1),
    #     'pagerank': (0.14285714285714274, 0.14285714285714296),
    # }
    #
    # # 选择当前指标的色带范围
    # vmin, vmax = colormap_ranges.get(value_col, (df[value_col].min(), df[value_col].max()))
    #
    # # 构造色带
    # # colormap = cm.linear.Inferno_09.scale(vmin, vmax)
    #
    # colormap = cm.linear.OrRd_09.scale(vmin, vmax)

    def get_colormap(value_col):
        colormap_ranges = {
            'degree': (0, 160),
            'clustering': (0, 1),
            'closeness': (0, 1),
            'pagerank': (0.14285714285714274, 0.14285714285714296),
        }
        vmin, vmax = colormap_ranges.get(value_col, (0, 1))
        return cm.linear.OrRd_09.scale(vmin, vmax), vmin, vmax

    colormap, vmin, vmax = get_colormap(value_col)

    colormap.caption = title  # 设置图例标题

    m = folium.Map(
        location=[30, 114],
        zoom_start=10,
        tiles='CartoDB.DarkMatter',
        control_scale=True,
        attr="Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
    )

    # 添加武汉边界
    json_file = 'data/420000.geojson'
    with open(json_file, 'r', encoding='utf-8') as f:
        wuhan_geojson = json.load(f)

    folium.GeoJson(wuhan_geojson,
                   style_function=lambda x: {'color': 'white', 'weight': 1.5},
                   name="武汉边界").add_to(m)

    for _, row in df.iterrows():
        bounds = [
            [row['Lat_Min'], row['Lon_Min']],
            [row['Lat_Min'], row['Lon_Max']],
            [row['Lat_Max'], row['Lon_Max']],
            [row['Lat_Max'], row['Lon_Min']],
            [row['Lat_Min'], row['Lon_Min']]
        ]
        val = row[value_col]

        folium.Polygon(
            locations=bounds,
            color='#666666',           # 灰色边界
            weight=0,
            fill=True,
            fill_color=colormap(val),
            fill_opacity=0.8
        ).add_to(m)
    # 手动添加图例 div（右下，略偏左，字体白色）
    # 添加图例（手动控制位置和颜色）
    # 设置标题 & 获取色带范围
    # 设置标题 & 获取色带范围
    colormap.caption = title
    min_val = vmin
    max_val = vmax

    # 提取渐变色列表（取10个颜色构造渐变）
    colors = colormap.colors  # 颜色列表
    if isinstance(colors[0], tuple):  # 有些版本返回RGB tuple
        from matplotlib.colors import to_hex
        colors = [to_hex(c) for c in colors]

    # 如果颜色数少于10个，则插值补齐
    if len(colors) < 10:
        gradient_colors = [colormap(min_val + i * (max_val - min_val) / 9) for i in range(10)]
    else:
        gradient_colors = colors[:10]

    gradient_css = ", ".join(gradient_colors)

    # 构造图例HTML（带色带 & 两端标签）
    legend_html = f'''
    <div style="position: fixed; 
         bottom: 20px; left: 65%; z-index:9999; 
         font-size:14px;
         background-color: rgba(250, 250, 250, 0.6);
         padding: 10px; border-radius: 5px; color: white;
         width: 460px;">

        <div style="text-align:center; margin-bottom:5px;">{title}</div>
        <div style="display: flex; align-items: center;">
            <div style="flex:1; text-align: left;">{min_val:.17f}</div>
            <div style="flex:6; height: 8px; 
                        background: linear-gradient(to right, {gradient_css}); 
                        margin: 0 10px;
                        border: 1px solid white;
                        border-radius: 2px;"></div>
            <div style="flex:1; text-align: right;">{max_val:.17f}</div>
        </div>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

    # colormap.add_to(m)  # 添加图例
    m.save(html_file)
    print(f"{title} 已保存为 {html_file}")

color='red'
# # plot_grid_with_values(metrics_df, 'degree', 'Degree', f'./result/bert/wh/128/{stage}/{color}/degree_map.html')
# plot_grid_with_values(metrics_df, 'pagerank', 'PageRank', f'./result/bert/wh/128/{stage}/{color}/pagerank_map.html')
# # plot_grid_with_values(metrics_df, 'closeness', 'Centrality Similarity', f'./result/bert/wh/128/{stage}/{color}/centrality_map.html')
# # plot_grid_with_values(metrics_df, 'clustering', 'Clustering Coefficient', f'./result/bert/wh/128/{stage}/{color}/clustering_map.html')
