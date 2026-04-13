import math
import pickle
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


# Read the stored adjacency matrix
def load_graphs_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        adjacency_matrices = pickle.load(f)
    return adjacency_matrices


# Converting an adjacency matrix to a NetworkX graph object
def convert_to_graphs(adjacency_matrices):
    graphs = [nx.from_numpy_array(matrix) for matrix in adjacency_matrices]
    return graphs

# Calculation chart features
def compute_graph_features(graph):
    degrees = [d for n, d in graph.degree()]#
    clustering_coeff = list(nx.clustering(graph).values())
    # eigenvector_centrality = list(nx.eigenvector_centrality_numpy(graph).values())#
    pagerank = list(nx.pagerank(graph).values())
    betweenness_centrality = list(nx.betweenness_centrality(graph).values())

    # Calculate the mean value of the feature
    features = [
        np.mean(degrees),
        np.mean(clustering_coeff),
        # np.mean(eigenvector_centrality),
        np.mean(pagerank),
        np.mean(betweenness_centrality)
    ]
    return features


# Estimating the optimal number of clusters using the elbow method
def estimate_optimal_clusters_elbow(features, max_clusters=8):
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()

    elbow_point = np.diff(sse, 2).argmin() + 2
    return elbow_point




# 主函数
def main1(file):
    # file='result/256'
    file_path = f'{file}/graphs.pkl'
    f = open(file_path, 'rb')
    data = pickle.load(f)
    adjacency_matrices = load_graphs_from_pkl(file_path)
    n = int(math.sqrt(len(adjacency_matrices)))
    # adjacency_matrices=[value for value in adjacency_matrices.values()]

    graphs = convert_to_graphs(adjacency_matrices)
    features = np.array([compute_graph_features(g) for g in graphs])

    # Standardised features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Estimating the optimal number of clusters using the elbow method
    optimal_clusters_elbow = estimate_optimal_clusters_elbow(scaled_features)
    print(f"Optimal number of clusters (Elbow Method): {optimal_clusters_elbow}")

    # 使用轮廓系数估计最优簇数量
    # optimal_clusters_silhouette = estimate_optimal_clusters_silhouette(scaled_features)
    # print(f"Optimal number of clusters (Silhouette Method): {optimal_clusters_silhouette}")

    n_clusters = optimal_clusters_elbow
    labels, kmeans = cluster_graphs(scaled_features, n_clusters)
    lable=labels.reshape(n,n)

    # Output clustering results
    for i, label in enumerate(labels):
        print(f"Graph {i} is in cluster {label}")


    result_df = pd.DataFrame({'Graph': list(range(len(labels))), 'Cluster': labels})
    with open(f'{file}/clustering_results.pkl', 'wb') as f:
        pickle.dump(lable, f)


def cluster_graphs(features, n_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    return labels, kmeans

if __name__=="__main__":
    main1('result/bert/wh/256')
