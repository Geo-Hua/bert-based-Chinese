import json
import math
import pickle
import webbrowser
import requests
import numpy as np
import folium
from scipy.ndimage import zoom
import pandas as pd
from caculate import caculate_index
class QuadTreeNode:
    def __init__(self, val=None, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None, x=None, y=None,
                 size=None):
        self.val = val  # value
        self.topLeft = topLeft  # upper-left
        self.topRight = topRight  # upper-right
        self.bottomLeft = bottomLeft  #
        self.bottomRight = bottomRight  #
        self.x = x  #
        self.y = y  #
        self.size = size  # size


def can_merge(grid, x, y, size):
    """Determine if a sub-matrix can be combined into a single value"""
    val = grid[x][y]
    for i in range(x, x + size):
        for j in range(y, y + size):
            if grid[i][j] != val:
                return False
    return True


def merge(grid, x, y, size, rectangles):
    """Recursive merging, building the quadtree stepwise upwards and recording the merged regions"""
    if size == 1 or can_merge(grid, x, y, size):
        # Leaf nodes or nodes that can be merged
        rectangles.append((x, y, size, grid[x][y]))  # Records consolidation area
        return QuadTreeNode(val=grid[x][y], x=x, y=y, size=size)

    mid = size // 2
    topLeft = merge(grid, x, y, mid, rectangles)
    topRight = merge(grid, x, y + mid, mid, rectangles)
    bottomLeft = merge(grid, x + mid, y, mid, rectangles)
    bottomRight = merge(grid, x + mid, y + mid, mid, rectangles)

    # Records consolidation area
    rectangles.append((x, y, size, None))

    return QuadTreeNode(topLeft=topLeft, topRight=topRight,
                        bottomLeft=bottomLeft, bottomRight=bottomRight, x=x, y=y, size=size)


def interpolate_grid(grid, target_size):
    """Interpolation to resize the grid to the target size"""
    original_size = len(grid)
    zoom_factor = target_size / original_size
    interpolated_grid = zoom(grid, zoom_factor, order=1)  # 使用最近邻插值
    return interpolated_grid


def plot_grid_on_map(grid, rectangles, color_map, n, output_file, min_lat, max_lat, min_lon, max_lon):
    """Plot the merged matrix area on a map and save it as an HTML file"""
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='CartoDB.DarkMatter',
        control_scale=True,
        attr="Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL."  # 添加版权声明
    )
    # wuhan_geojson = requests.get('https://geojson.cn/api/china/420000.json').json()
    json_file='../data/420000.geojson'
    with open(json_file, 'r', encoding='utf-8') as f:
        wuhan_geojson = json.load(f)

    folium.GeoJson(wuhan_geojson,
                   style_function=lambda x: {'color': 'white', 'weight': 1.5},
                   name="武汉边界").add_to(m)

    # Calculate the actual latitude and longitude range of each rectangular area
    lat_step = (max_lat - min_lat) / n
    lon_step = (max_lon - min_lon) / n


    rectangles.sort(key=lambda rect: rect[2],reverse=True)
    # Add rectangular overlays to each matrix region
    for rect in rectangles:
        x, y, size, value = rect
        # choose color
        color = color_map.get(size, 'rgba(255,255,255,0)')

        # Calculate the actual latitude and longitude range of the rectangular area
        lat1 = min_lat + y * lat_step
        lat2 = min_lat + (y + size) * lat_step
        lon1 = min_lon + x * lon_step
        lon2 = min_lon + (x + size) * lon_step

        # Adding rectangular areas to the map
        folium.Rectangle(
            bounds=[(lat1, lon1), (lat2, lon2)],
            color='black',
            fill=True,
            weight=0.5,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)


    # save
    m.save(output_file)


def construct_quad_tree(grid):
    n = len(grid)
    rectangles = []
    quad_tree = merge(grid, 0, 0, n, rectangles)
    return quad_tree, rectangles

def qt(file,num):
    # file='result/256'
    # load data
    with open(f'{file}/clustering_results.pkl', 'rb') as f:
        grid = pickle.load(f)

    # Define target size
    target_size = int(math.sqrt(grid.size))

    # Interpolate the grid to populate it
    interpolated_grid = interpolate_grid(grid, target_size)

    # Define a fixed list of colours
    size_color_map = {
        1: 'rgba(78,171,144,0.6)',  # 1x1
        2: 'rgba(237,221,195,0.6)',  # 2x2
        4: 'rgba(238,191,109,0.6)',  # 4x4
        16: 'rgba(217,79,51,0.6)',  # 8x8
        32: 'rgba(131,64,38,0.6)',  # 16x16
        8:'rgba(142,182,156,0.6)',# 32x32

    }

    # Construct a quadtree and get the region of each merge
    quad_tree, rectangles = construct_quad_tree(interpolated_grid)
    caculate_index(interpolated_grid,num)
    data = pd.read_csv(f'{file}/grid_lat_lon.csv')  # Refined grid

    # Define latitude and longitude ranges
    min_lat = min(data['Lat_Max'])
    max_lat = max(data['Lat_Min'])
    min_lon = min(data['Lon_Max'])
    max_lon = max(data['Lon_Min'])

    # Plot the matrix with boundaries and save it as an HTML file
    plot_grid_on_map(interpolated_grid, rectangles, size_color_map, target_size, f'{file}/dark_before_13.html', min_lat, max_lat, min_lon, max_lon)


if __name__=="__main__":
    file='result/bert/wh/256'
    size=256
    qt(file,size)