from idlelib.iomenu import encoding

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os

# 1. 读取街道边界数据
streets = gpd.read_file('../result/bert/wh/Street_index/L4/China_Wuhan, HUB_L4.shp')
# streets = gpd.read_file('../result/bert/sh/Street_index/L1/China_Shanghai, SHG-JS-ZJ_L1.shp')
if streets.crs != "EPSG:4326":
    streets = streets.to_crs(epsg=4326)

# 可选：查看字段名（确认用到的是 'NAME' 和 'DISTRICT'）
print(streets.columns)
stage="pos-neg/L3/before"
# 2. 读取微博数据
with open(f'../result/bert/wh/before_2020_02_12_new.csv', 'r', encoding='utf-8', errors='ignore') as f:
# with open('../result/bert/sh/comments_data_with_emotions.csv', 'r', encoding='utf-8', errors='ignore') as f:
    weibo_df = pd.read_csv(f)
weibo_df = weibo_df.dropna(subset=['lon', 'lat'])

# 转为GeoDataFrame
weibo_df['geometry'] = weibo_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
weibo_gdf = gpd.GeoDataFrame(weibo_df, geometry='geometry', crs='EPSG:4326')

# 3. 空间连接：微博 → 所属街道
streets_gdf = streets[['geometry', 'index']]  # 改成实际字段名
weibo_with_street = gpd.sjoin(weibo_gdf, streets_gdf, how='inner', predicate='within')

# 4. 输出完整的匹配结果（每条微博 → 所属街道）
weibo_with_street.to_csv(f'../result/bert/wh/Street_index/{stage}/all_weibo_with_index.csv', index=False,encoding='utf-8-sig')
# weibo_with_street.to_csv('../result/bert/sh/Street_index/L1/all_weibo_with_index.csv', index=False,encoding='utf-8-sig')

# 5. 按街道聚合，分别输出微博数据到单独文件
output_dir = f'../result/bert/wh/Street_index/{stage}/output_weibo_by_index'
# output_dir = '../result/bert/sh/Street_index/L1/output_weibo_by_index'
os.makedirs(output_dir, exist_ok=True)

# 用 DISTRICT（区）和 NAME（街道）字段分组
grouped = weibo_with_street.groupby(['index'])

for index, group in grouped:
    # 构建文件名，例如“黄浦区_打浦桥街道.csv”
    filename = f"{index}.csv".replace("/", "_").replace("\\", "_")
    filepath = os.path.join(output_dir, filename)
    group.to_csv(filepath, index=False,encoding='utf-8-sig')

print("✅ 所有微博数据已按街道分类输出完毕。")
