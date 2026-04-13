import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 读取街道边界数据并统一投影
gdf_street = gpd.read_file("./data/boundary/area.shp")
gdf_street = gdf_street.to_crs("EPSG:32651")

# 读取情绪预测结果
emotions = pd.read_csv("./result/bert/wh/emotion_prediction_wh.csv")
geometry = [Point(xy) for xy in zip(emotions["lon"], emotions["lat"])]
gdf_emo = gpd.GeoDataFrame(emotions, geometry=geometry, crs="EPSG:4326").to_crs(gdf_street.crs)

# 添加情绪类型字段
def classify_emotion(cate):
    if cate == 100:
        return 'Compound Emotion'
    elif str(cate).startswith('1'):
        return 'Single Emotion'
    else:
        return 'Dominant-Subordinate Emotion'

gdf_emo["emotion_type"] = gdf_emo["cate"].apply(classify_emotion)

# 空间连接，确定每条评论落在哪个街道
emo_join = gpd.sjoin(gdf_emo, gdf_street, predicate="within")

# 分组计数：每个街道每类情绪数量
emo_count = emo_join.groupby(["乡", "emotion_type"]).size().unstack(fill_value=0)

# 计算每个街道总评论数
emo_count["总数"] = emo_count.sum(axis=1)

# 计算各类情绪所占比例
for col in emo_count.columns:
    if col != "总数":
        emo_count[col] = emo_count[col] / emo_count["总数"]

# 去除总数列，并重置索引
emo_ratio = emo_count.drop(columns=["总数"]).reset_index()

# 合并回街道图层
gdf_street = gdf_street.merge(emo_ratio, on="乡", how="left")

# 仅导出乡与比例数据列
emotion_columns = ["Single Emotion", "Dominant-Subordinate Emotion", "Compound Emotion"]
existing_columns = [col for col in emotion_columns if col in gdf_street.columns]
columns_to_save = ['乡'] + existing_columns
gdf_street[columns_to_save].to_csv("./result/多因子/emotion_ratio.csv", index=False)
