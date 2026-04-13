import pickle
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# === 读取pkl中的矩阵和街道名 ===
def load_street_graphs(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    index = data['index']
    matrices = data['matrices']
    return index, matrices


# === 将邻接矩阵转为图对象 ===
def convert_to_graphs(matrices):
    return [nx.from_numpy_array(mat) for mat in matrices]


# === 提取图的结构特征（均值）===
def compute_graph_features(graph):
    degrees = [d for _, d in graph.degree()]
    clustering_coeff = list(nx.clustering(graph).values())
    pagerank = list(nx.pagerank(graph).values())
    betweenness = list(nx.betweenness_centrality(graph).values())

    return {
        "degree": np.mean(degrees),
        "clustering": np.mean(clustering_coeff),
        "pagerank": np.mean(pagerank),
        "betweenness": np.mean(betweenness)
    }


# === 肘部法估计聚类数 ===
def estimate_optimal_clusters_elbow(features, max_clusters=8):
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    elbow_point = np.diff(sse, 2).argmin() + 2
    return elbow_point


# === 聚类主函数 ===
def main(file):
    input_pkl = f"{file}/graphs.pkl"
    output_pkl = f"{file}/clustering_results.pkl"
    output_csv = f"{file}/graph_features.csv"

    # 加载数据
    index, matrices = load_street_graphs(input_pkl)
    graphs = convert_to_graphs(matrices)

    # 特征提取
    feature_dicts = [compute_graph_features(g) for g in graphs]
    features = np.array([[fd["degree"], fd["clustering"], fd["pagerank"], fd["betweenness"]]
                         for fd in feature_dicts])

    # === 保存指标到 CSV ===
    df_features = pd.DataFrame(feature_dicts, index=index)
    df_features.index.name = "index"
    df_features.to_csv(output_csv, encoding="utf-8-sig")
    print(f"Graph features saved to {output_csv}")

    # 聚类
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    optimal_k = estimate_optimal_clusters_elbow(scaled_features)
    print(f"Optimal cluster count (elbow): {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    # 组装结果：{街道: 类别}
    clustering_result = {idx: int(label) for idx, label in zip(index, labels)}

    # 保存结果
    with open(output_pkl, 'wb') as f:
        pickle.dump(clustering_result, f)

    print(f"Clustering results saved to {output_pkl}")


# === 执行入口 ===
if __name__ == "__main__":
    file = '../result/bert/wh/Street_index/L4'  # 可根据实际路径修改
    # file = '../result/bert/sh/Street_index/pos-neg/L1'  # 可根据实际路径修改
    main(file)
