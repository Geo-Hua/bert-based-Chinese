import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
csv_path = "./merged.csv"
df = pd.read_csv(csv_path)

# 输出路径
output_all = "./output/degree_boxplots"
os.makedirs(output_all, exist_ok=True)

# 情绪与指标字段
emotion_cols = ["Single Emotion", "Dominant-Subordinate Emotion", "Compound Emotion"]
extra_cols = ["degree", "avg_clustering"]   # 👈 新增的两列
exclude_cols = emotion_cols + extra_cols + ['乡']
indicator_cols = [col for col in df.columns if col not in exclude_cols]

for indicator in indicator_cols:
    if not np.issubdtype(df[indicator].dtype, np.number):
        continue

    # 箱型图过滤离群值
    Q1 = df[indicator].quantile(0.25)
    Q3 = df[indicator].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[(df[indicator] >= Q1 - 1.5 * IQR) & (df[indicator] <= Q3 + 1.5 * IQR)]

    # 分箱
    bin_width = (Q3 - Q1) / 5 if (Q3 - Q1) > 0 else 0.1
    bins = np.arange(df_filtered[indicator].min(), df_filtered[indicator].max() + bin_width, bin_width)
    bins = np.unique(bins)
    bin_labels = [f"{round(bins[i], 2)}~{round(bins[i + 1],2)}" for i in range(len(bins) - 1)]

    df_filtered["indicator_bin"] = pd.cut(
        df_filtered[indicator],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
        ordered=False
    )

    # 转换成长格式
    df_long = df_filtered.melt(
        id_vars=["indicator_bin"],
        value_vars=emotion_cols,
        var_name="Emotion Type",
        value_name="Emotion Ratio"
    )
    df_long = df_long[df_long["Emotion Ratio"] > 0]

    if df_long.empty:
        continue

    # 每种情绪在不同bin下的平均值
    mean_ratios = df_long.groupby(["indicator_bin", "Emotion Type"])["Emotion Ratio"].mean().reset_index()

    # 每个bin的 degree 平均值（用于右轴柱状图）
    df_degree = df_filtered.groupby("indicator_bin")["degree"].mean().reset_index()

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 画箱型图
    sns.boxplot(x="indicator_bin", y="Emotion Ratio", hue="Emotion Type",
                data=df_long, palette="Set2", width=0.6, ax=ax1)
    ax1.set_xlabel(f"{indicator} Interval")
    ax1.set_ylabel("Emotion Ratio")
    # ax1.set_title(f"Emotion Ratio vs {indicator}", fontsize=14)
    ax1.tick_params(axis='x', rotation=45)

    # 在箱型图上加柔和灰线（每种情绪的平均值）
    for emo in emotion_cols:
        sub_mean = mean_ratios[mean_ratios["Emotion Type"] == emo]
        ax1.plot(sub_mean["indicator_bin"], sub_mean["Emotion Ratio"],
                 linestyle="--", linewidth=1.5, alpha=0.7, label=f"{emo} Mean")

    # 添加 degree 柱状图（右轴）
    ax2 = ax1.twinx()
    ax2.bar(df_degree["indicator_bin"], df_degree["degree"],
            alpha=0.3, color="steelblue", label="Degree")
    ax2.set_ylabel("Degree")

    # 合并图例
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2, title="Legend", bbox_to_anchor=(1, 1), loc="lower left")
    # ❌ 去掉所有图例
    ax1.get_legend().remove()
    # ax2 没有图例，不需要处理
    plt.tight_layout()
    filename = f"{indicator}_with_degree.png".replace("/", "_")
    plt.savefig(os.path.join(output_all, filename), dpi=300)
    plt.close()