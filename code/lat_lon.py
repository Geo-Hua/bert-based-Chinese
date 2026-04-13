import numpy as np
import pandas as pd

def generate_grid(lat_min, lat_max, lon_min, lon_max, grid_size=128):
    # Calculate original latitude and longitude ranges
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # Calculate the new latitude and longitude ranges so that each grid is the same size
    lat_step = lat_range / grid_size
    lon_step = lon_range / grid_size

    # Extended latitude and longitude ranges to ensure consistent size of each grid
    new_lat_range = lat_step * grid_size
    new_lon_range = lon_step * grid_size

    # Updating the maximum values for latitude and longitude
    new_lat_max = lat_min + new_lat_range
    new_lon_max = lon_min + new_lon_range

    # Creating grids
    grid_data = []
    grid_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            lat_start = lat_min + i * lat_step
            lat_end = lat_min + (i + 1) * lat_step if i + 1 < grid_size else new_lat_max
            lon_start = lon_min + j * lon_step
            lon_end = lon_min + (j + 1) * lon_step if j + 1 < grid_size else new_lon_max
            grid_data.append([grid_id, lat_start, lat_end, lon_start, lon_end])
            grid_id += 1

    return grid_data, new_lat_max, new_lon_max

def save_grid_to_csv(grid_data, filename):
    df = pd.DataFrame(grid_data, columns=["Grid_ID", "Lat_Min", "Lat_Max", "Lon_Min", "Lon_Max"])
    df.to_csv(filename, index=False,encoding='utf-8-sig')

def latlon(input_file,output_file,size):
    # file_path='data/wh_data_cleaned.csv'
    file_path=input_file

    column_to_read=['created_at','content','lon','lat']
    reviews = pd.read_csv(file_path,usecols=column_to_read)

    # Get latitude and longitude ranges
    lat = reviews['lat']
    lon = reviews['lon']
    lat_min = min(lat)
    lat_max = max(lat)
    lon_min = min(lon)
    lon_max = max(lon)

    file=output_file

    grid_data, new_lat_max, new_lon_max = generate_grid(lat_min, lat_max, lon_min, lon_max, grid_size=size)

    # output
    save_grid_to_csv(grid_data, f"{file}/grid_lat_lon.csv")

    print(f"Grid has been divided into 128x128 grids, with the new lat_max: {new_lat_max} and lon_max: {new_lon_max}.")
#file='../result/bert/wh/128'
if __name__=="__main__":
    file = '../result/bert/wh/256'
    size = 256
    latlon('../data/emotion_prediction_wh.csv',file,size)