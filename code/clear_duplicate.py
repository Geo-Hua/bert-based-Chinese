import pandas as pd
import re

# Define a function to remove HTML tags
def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)  # 使用正则表达式删除HTML标签
    return clean_text


def  cd(input_file,out_file):
    # input CSV file
    df = pd.read_csv(input_file)

    df['content'] = df['content'].apply(remove_html_tags)

    # Determine if there are duplicate rows and remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Save the processed CSV file
    df_cleaned.to_csv(out_file, index=False,encoding='utf-8-sig')

    print("重复行已去除，清理后的文件已保存为 'wh_data_cleaned.csv'.")

if __name__=="__main__":
    cd('../data/wh_data.csv','../data/wh_data_cleaned.csv')
