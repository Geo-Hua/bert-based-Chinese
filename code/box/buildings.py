import geopandas as gpd
import pandas as pd

# 1. 读取行政区（含 AREA 字段）
districts = gpd.read_file("./data/boundary/area.shp")
districts = districts.dissolve(by="乡", as_index=False)
districts = districts.to_crs("EPSG:3857")
districts['area_m2'] = districts.geometry.area

# 2. 读取建筑物数据（含 Height 字段）
buildings = gpd.read_file("./data/building/hubei.shp")
buildings = buildings.to_crs("EPSG:3857")
buildings = buildings[buildings['Height'].notnull()]
buildings['footprint_area'] = buildings.geometry.area
buildings['volume'] = buildings['Height'] * buildings['footprint_area']
buildings['is_highrise'] = buildings['Height'] > 60  # 超过60米定义为高层

# 3. 空间连接建筑物与行政区（每个建筑物归属一个行政区）
buildings_with_area = gpd.sjoin(buildings, districts[['乡', 'geometry']], how='inner', predicate='intersects')

# 4. 分组统计建筑物指标
grouped = buildings_with_area.groupby('乡').agg({
    'Height': ['mean', 'max'],
    'footprint_area': 'sum',
    'volume': 'sum',
    'is_highrise': 'sum'
}).reset_index()

grouped.columns = ['乡', 'height_mean', 'height_max', 'building_area', 'building_volume', 'highrise_count']

# 5. 合并行政区面积，计算密度指标
result = grouped.merge(districts[['乡', 'area_m2']], on='乡')
result['building_density'] = result['building_area'] / result['area_m2']
result['volume_density'] = result['building_volume'] / result['area_m2']
result['highrise_ratio'] = result['highrise_count'] / buildings_with_area.groupby('乡').size().values

# 6. 保存输出
result.to_csv("./result/多因子/building_height_by_district.csv", index=False)
