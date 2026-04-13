import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import re
import folium
import json
from folium.features import GeoJsonTooltip
matplotlib.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
stage="pos-neg/L4"
# 参数配置
clustering_pkl_path = f'../result/bert/wh/Street_index/{stage}/clustering_results.pkl'
street_shapefile_path = '../result/bert/wh/Street_index/L4/China_Wuhan, HUB_L4.shp'
# clustering_pkl_path = '../result/bert/sh/Street_index/pos-neg/L1/clustering_results.pkl'
# street_shapefile_path = '../result/bert/sh/Street_index/pos-neg/L1/China_Shanghai, SHG-JS-ZJ_L1.shp'  # 替换为你的街道边界Shapefile路径
street_name_field = 'index'  # shp中的街道字段名，应为纯“街道名”
title = "街道聚类结果图"
output_fig = f'../result/bert/wh/Street_index/{stage}/cluster_map.png'
# output_fig = '../result/bert/sh/Street_index/pos-neg/L1/cluster_map.png'
output_html = f'../result/bert/wh/Street_index/{stage}/cluster_map.html'
# === Step 1: 读取聚类结果，并裁剪键为“街道名”部分 ===
with open(clustering_pkl_path, 'rb') as f:
    raw_cluster_dict = pickle.load(f)

# 仅保留下划线前的“街道名”部分
# 提取键中的数字部分，作为整数型 index
cleaned_cluster_dict = {}

for k, v in raw_cluster_dict.items():
    match = re.search(r'\d+', str(k))
    if match:
        index = int(match.group())
        cleaned_cluster_dict[index] = v

# === Step 2: 读取街道边界数据 ===
gdf = gpd.read_file(street_shapefile_path)

# === Step 3: 赋值聚类结果（按街道名匹配） ===
gdf['cluster'] = gdf[street_name_field].map(cleaned_cluster_dict)

# 检查匹配成功情况
matched = gdf['cluster'].notna().sum()
total = len(gdf)
print(f"成功匹配聚类类别的街道数: {matched} / {total}")

# 设定颜色映射
color_map = {
    0: '#7c7cba',  # 类别 0 - purple
    1: '#3fa0c0',  # 类别 1 - 橙色
    2: '#d5d9e5',  # 类别 2 - 绿色
    3:'#d6e0c8',
    4:'#fedec5',
    5:'#f0a19a',
    6:'#c7b8bd',
    7:'#fff0bc',
    8:'#00a664'
}
default_color = '#D3D3D3'  # 未分类 - 灰色

# 生成颜色列：使用 map 并填充 NaN 为灰色
gdf['color'] = gdf['cluster'].map(color_map).fillna(default_color)

# 绘图
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(ax=ax, color=gdf['color'], edgecolor='gray', linewidth=0.5)

# 添加文本标签
# for idx, row in gdf.iterrows():
#     centroid = row.geometry.centroid
#     cluster_label = row['cluster']
#     if pd.notna(cluster_label):
#         ax.text(centroid.x, centroid.y, int(cluster_label), fontsize=8, ha='center', va='center')

# 图例补充：添加自定义类别和灰色未分类

# === Step X: 图例补充：仅显示实际出现的类别 ===
# 统计当前GeoDataFrame中出现的聚类类别
actual_clusters = gdf['cluster'].dropna().unique().astype(int)

# 构建图例，仅显示实际存在的类别
legend_patches = [
    mpatches.Patch(color=color_map[k], label=f"category {k}")
    for k in sorted(actual_clusters) if k in color_map
]

# 添加“未分类”灰色图例项
legend_patches.append(mpatches.Patch(color=default_color, label="No Data"))

# 添加图例到图上
ax.legend(
    handles=legend_patches,
    loc='lower right',
    bbox_to_anchor=(1.0, 0.0),
    borderaxespad=0.2,
    title='Legend',
    fontsize=10,
    title_fontsize=12,
    frameon=True  # 可选：添加图例边框，提高可读性
)


# 图标题及美化
plt.title("街道聚类结果图", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
# === Step 6: 可选保存输出 ===
fig.savefig(output_fig, dpi=300)

# === Step 5: Folium 交互地图输出 ===
center_lat = gdf.geometry.centroid.y.mean()
center_lon = gdf.geometry.centroid.x.mean()
m = folium.Map(location=[center_lat, center_lon],  zoom_start=10,
        tiles='CartoDB.DarkMatter',
        control_scale=True,
        attr="Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.")

with open('../data/420000.geojson', 'r', encoding='utf-8') as f:
    wuhan_geojson = json.load(f)
    # 添加武汉边界
# folium.GeoJson(
#         wuhan_geojson,
#         style_function=lambda x: {'color': 'white', 'weight': 1.5},
#         name="武汉边界"
#     ).add_to(m )

def style_function(feature):
    cluster_val = feature['properties'].get('cluster')
    if cluster_val is not None and cluster_val in color_map:
        return {'fillColor': color_map[cluster_val],
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.9}
    else:
        return {'fillColor': default_color,
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.9}

folium.GeoJson(
    gdf,
    style_function=style_function,
    tooltip=GeoJsonTooltip(fields=[street_name_field, 'cluster'],
                           aliases=["街道", "聚类类别"],
                           localize=True)
).add_to(m)

# 添加自定义图例
legend_html = """
<div style="
position: fixed;
bottom: 20px;
right: 20px;
width: 130px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding: 10px;">
<b>Legend</b><br>
"""
for k in sorted(actual_clusters):
    legend_html += f'<i style="background:{color_map[k]};width:18px;height:18px;float:left;margin-right:10px;opacity:0.7;"></i> {k}<br>'
legend_html += f'<i style="background:{default_color};width:18px;height:18px;float:left;margin-right:10px;opacity:0.7;"></i> No Data<br>'
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

m.save(output_html)
print(f"HTML 地图已保存到: {output_html}")