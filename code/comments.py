import pandas as pd
import os

def comment(file,emotion_file):

    # Reading subdivided grid and raw grid data
    data = pd.read_csv(f'{file}/grid_lat_lon.csv')  # 细化后的格网

    # Read comment data
    comments_data = pd.read_csv(emotion_file)  # 评论数据文件

    # Ensure that the grid data has been sorted by latitude and longitude columns to speed up searches
    data_sorted = data.sort_values(by=['Lon_Min', 'Lat_Min'])

    # Batch processing of latitude/longitude interval matches using a function
    def get_grid_id(lon, lat, grid_data):
        # Get the Grid_ID by looking up the latitude/longitude interval.
        match = grid_data[(grid_data['Lon_Min'] <= lon) & (grid_data['Lon_Max'] > lon) &
                           (grid_data['Lat_Min'] <= lat) & (grid_data['Lat_Max'] > lat)]
        return match['Grid_ID'].iloc[0] if not match.empty else None

    # Converting Grid Data to DataFrame Indexes for Faster Lookups
    grid_index = pd.MultiIndex.from_frame(data[['Lon_Min', 'Lon_Max', 'Lat_Min', 'Lat_Max']], names=['Lon_Min', 'Lon_Max', 'Lat_Min', 'Lat_Max'])

    # 为评论数据添加 Grid_ID 列（细分格网）
    comments_data['Grid_ID'] = comments_data.apply(
        lambda row: get_grid_id(row['lon'], row['lat'], data_sorted), axis=1
    )

    comments_data = comments_data.dropna(subset=['Grid_ID'])

    df_xi = pd.DataFrame(comments_data)

    grouped_xi = df_xi.groupby('Grid_ID')

    # Determine if a folder exists
    output = f'{file}/result'
    if not os.path.exists(output):
        os.makedirs(output)

    # Iterate through each group and save it as a separate CSV file
    for group_name, group_data in grouped_xi:
        output_file = f'{output}/output_{group_name}.csv'
        group_data.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("原始格网内的评论已保存")



if __name__=="__main__":
    file='../result/wh/256'
    comment(file,'../data/emotion_prediction_wh.csv')