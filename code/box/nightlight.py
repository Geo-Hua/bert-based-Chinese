import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import pandas as pd

# 1. 打开栅格并获取 CRS
raster_path = "./data/nightlight/hubei.tif"
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

# 2. 读取矢量并转为栅格坐标系
streets = gpd.read_file("./data/boundary/China_Wuhan, HUB_L4.shp")
streets = streets.to_crs(raster_crs)

# 3. 按 AREA dissolve 为行政区
districts = streets.dissolve(by='index', as_index=False)
districts['geometry'] = districts['geometry'].buffer(0)  # 修复几何错误
districts['area_m2'] = districts.geometry.area

# 4. 逐区提取亮度值
results = []
with rasterio.open(raster_path) as src:
    for idx, row in districts.iterrows():
        geom = [mapping(row['geometry'])]
        out_image, _ = mask(src, geom, crop=True)
        data = out_image[0]
        valid_data = data[data != src.nodata]

        if valid_data.size > 0:
            stats = {
                'index': row['index'],
                'light_mean': float(np.mean(valid_data)),
                'light_std': float(np.std(valid_data)),
                'light_max': float(np.max(valid_data)),
                'light_sum': float(np.sum(valid_data)),
                'count': int(valid_data.size)
            }
        else:
            stats = {
                'index': row['index'],
                'light_mean': np.nan,
                'light_std': np.nan,
                'light_max': np.nan,
                'light_sum': np.nan,
                'count': 0
            }
        results.append(stats)

# 5. 转为 DataFrame 并计算密度
df_result = pd.DataFrame(results)
df_result = df_result.merge(districts[['index', 'area_m2']], on='index', how='left')
df_result['light_density'] = df_result['light_sum'] / df_result['area_m2']
df_result['light_density_km2'] = df_result['light_density'] * 1_000_000

# 6. 保存输出
df_result.to_csv("./多因子/nightlight_by_district.csv", index=False)
