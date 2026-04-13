import ast
import os
import pandas as pd

# 文件夹路径
folder_path = r"./result/bert/wh/L4/output_weibo_by_index/"  # 你的csv文件夹路径

# 三类情绪统计
emotion_class_count = {
    "单一情绪": 0,
    "复合情绪": 0,
    "主导附属情绪": 0
}

# 积极/消极/中性统计
pos_neg_count = {
    "积极": 0,
    "消极": 0,
    "中性": 0
}

# 保存文件结果
file_results = []

for file_name in os.listdir(folder_path):
    if not file_name.lower().endswith(".csv"):
        continue

    file_path = os.path.join(folder_path, file_name)

    try:
        df = pd.read_csv(file_path)

        if df.empty or "cate" not in df.columns or "pos-neg" not in df.columns:
            continue

        # ===== 情绪类别判断 =====
        single_count = df["cate"].astype(str).str.startswith("1").sum() - (df["cate"] == 100).sum()
        multi_count = (df["cate"] == 100).sum()
        dom_sub_count = len(df) - single_count - multi_count

        counts = {
            "单一情绪": single_count,
            "复合情绪": multi_count,
            "主导附属情绪": dom_sub_count
        }
        max_type = max(counts, key=counts.get)

        if counts[max_type] == 0:
            continue

        emotion_class_count[max_type] += 1

        # ===== 每行判断积极/消极/中性 =====
        pos_rows = 0
        neg_rows = 0
        neutral_rows = 0

        for val in df["pos-neg"]:
            try:
                d = ast.literal_eval(str(val))
                if isinstance(d, dict) and "positive" in d and "negative" in d:
                    pos_val = float(d["positive"])
                    neg_val = float(d["negative"])
                    if abs(pos_val - neg_val) < 0.05:
                        neutral_rows += 1
                    elif pos_val > neg_val:
                        pos_rows += 1
                    else:
                        neg_rows += 1
            except Exception:
                continue

        # 决定文件的整体倾向
        row_counts = {
            "积极": pos_rows,
            "消极": neg_rows,
            "中性": neutral_rows
        }
        sentiment_type = max(row_counts, key=row_counts.get)

        if row_counts[sentiment_type] == 0:
            continue

        pos_neg_count[sentiment_type] += 1

        # 保存文件结果
        file_results.append({
            "文件名": file_name,
            "情绪类别": max_type,
            "积极/消极/中性": sentiment_type,
            "积极行数": pos_rows,
            "消极行数": neg_rows,
            "中性行数": neutral_rows
        })

    except Exception as e:
        print(f"读取 {file_name} 出错: {e}")

# ===== 占比计算 =====
total_emotion_files = sum(emotion_class_count.values())
emotion_ratio = {k: v / total_emotion_files for k, v in emotion_class_count.items()} if total_emotion_files > 0 else {}

total_sentiment_files = sum(pos_neg_count.values())
pos_neg_ratio = {k: v / total_sentiment_files for k, v in pos_neg_count.items()} if total_sentiment_files > 0 else {}

# ===== 输出结果 =====
print("情绪类别数量:", emotion_class_count)
print("情绪类别占比:", emotion_ratio)
print("积极/消极/中性数量:", pos_neg_count)
print("积极/消极/中性占比:", pos_neg_ratio)