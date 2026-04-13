import os
import pandas as pd

# 设置CSV文件所在的文件夹路径
folder_path = "./result/bert/wh/多因子"

# 获取文件夹中所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 初始化主DataFrame（使用第一个文件作为基准）
main_df = None

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件失败：{file}，错误：{e}")
        continue

    # 检查是否包含“乡”列
    if '乡' not in df.columns:
        print(f"文件 {file} 不包含 '乡' 列，跳过。")
        continue

    # 如果是第一个文件，则作为主df
    if main_df is None:
        main_df = df
        continue

    # 查找匹配的乡名
    matched = df['乡'].isin(main_df['乡'])

    if matched.sum() == 0:
        unmatched乡 = df['乡'].unique()
        print(f"文件 {file} 中无匹配 '乡'，包含以下未匹配项：{list(unmatched乡)}，已跳过。")
        continue

    # 只保留与主df匹配的乡
    df = df[df['乡'].isin(main_df['乡'])]

    # 按“乡”列合并
    main_df = pd.merge(main_df, df, on='乡', how='left')

# 输出合并后的数据
output_path = os.path.join(folder_path, './merged_result1.csv')
main_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"合并完成，结果已保存为：{output_path}")
