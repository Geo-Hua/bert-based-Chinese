
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np

# === 1. 读取行政区矢量边界 ===
gdf = gpd.read_file("./data/boundary/area.shp")

# 读取土地覆盖栅格，统一投影
with rasterio.open("./data/landuse/hubei.tif") as src:
    raster_crs = src.crs

gdf = gdf.to_crs(raster_crs)
gdf['geometry'] = gdf['geometry'].buffer(0)  # 修复无效几何

# dissolve 为行政区
districts = gdf.dissolve(by='乡', as_index=False)

# === 2. zonal_stats 提取各行政区内土地覆盖像素数量 ===
landcover_path = "./data/landuse/hubei.tif"

stats = zonal_stats(
    vectors=districts,
    raster=landcover_path,
    categorical=True,
    geojson_out=True,
    nodata=0  # 或设为 None 看实际数据情况
)

# === 3. 转为 DataFrame，提取各行政区 Value 类型的像素数量 ===
results = []
for feature in stats:
    area = feature['properties']['乡']
    counts = {int(k): v for k, v in feature['properties'].items() if isinstance(k, int) or k.isdigit()}
    counts['乡'] = area
    results.append(counts)

df = pd.DataFrame(results)
df = df.fillna(0)

# === 4. 计算总像素、各类占比与面积（30m 分辨率示例） ===
pixel_size = 30
pixel_area = pixel_size ** 2  # 单位 m²

df['total_pixel'] = df.drop(columns='乡').sum(axis=1)

for col in df.columns:
    if col not in ['乡', 'total_pixel']:
        df[f'{col}_pct'] = df[col] / df['total_pixel']
        df[f'{col}_area_m2'] = df[col] * pixel_area

# === 5. 保存结果 ===
df.to_csv("./result/多因子/landcover_by_district.csv", index=False)
