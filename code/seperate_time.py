import pandas as pd

def time(input_file,output_before,output_after):
    # read csv
    df = pd.read_csv(input_file)

    # Convert Time Format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%Y/%m/%d %H:%M')

    # Defining the split date
    split_date = pd.to_datetime('2020/2/12', format='%Y/%m/%d')

    # Segmentation of data by time
    before_split = df[df['created_at'] < split_date]
    after_split = df[df['created_at'] >= split_date]

    # save
    before_split.to_csv(output_before, index=False,encoding='utf-8-sig')
    after_split.to_csv(output_after, index=False,encoding='utf-8-sig')

if __name__=="__main__":
    time('emotion_prediction_wh.csv','result/bert/wh/128/before/before_2020_02_12_new.csv','result/bert/wh/128/after/after_2020_02_12_new.csv')
