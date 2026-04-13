
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


def caculate_index(grid,num):
    # Counting the size of each clustered block
    def find_clusters(grid):
        visited = np.zeros_like(grid, dtype=bool)
        clusters = []  # Record the dimensions of all grids

        def bfs(x, y, label):
            """Use BFS to find the size of the entire grid area"""
            queue = [(x, y)]
            visited[x, y] = True
            cluster_cells = [(x, y)]  # Record all cells of the cluster

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四邻域搜索
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if not visited[nx, ny] and grid[nx, ny] == label:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                            cluster_cells.append((nx, ny))

            return cluster_cells

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 0 and not visited[i, j]:  # 忽略 0（未聚类区域）
                    cluster_size = len(bfs(i, j, grid[i, j]))
                    clusters.append(cluster_size)

        return clusters

    clusters = find_clusters(grid)

    # Calculate the final number of meshes
    num_final_grids = len(clusters)
    print(f"聚类后格网数目: {num_final_grids}")

    #Calculate the average grid size
    # The average size of the grid can be measured in terms of area or side lengths
    avg_grid_area = np.mean(clusters)
    avg_grid_length = np.mean([np.sqrt(size) for size in clusters])

    print(f"聚类后格网平均面积: {avg_grid_area:.2f} 个单元")
    print(f"聚类后格网平均边长: {avg_grid_length:.2f} 单元")


    # Calculation of Moran's I spatial autocorrelation index
    def morans_I(grid):
        valid_cells = [(x, y, grid[x, y]) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] > 0]
        if len(valid_cells) < 2:
            return None

        df = pd.DataFrame(valid_cells, columns=["x", "y", "value"])
        X = df["value"].values
        W = squareform(pdist(df[["x", "y"]]))  # Calculate the Euclidean distance
        W = np.exp(-W)  # normalised weight
        W /= W.sum()  # normalisation

        X_mean = np.mean(X)
        num = np.sum(W * ((X - X_mean)[:, None] * (X - X_mean)))
        den = np.sum((X - X_mean) ** 2)

        return num / den


    morans_I_value = morans_I(grid)
    print(f"\nMoran’s I 空间自相关指数: {morans_I_value:.4f}")

    unique_classes, class_counts = np.unique(grid, return_counts=True)
    class_dict = dict(zip(unique_classes, class_counts))
    num_grids = sum(class_counts)  # Number of non-space nets
    class_ratios = {k: v / num_grids for k, v in class_dict.items()}
    print("\n类别占比:")
    for cls, ratio in class_ratios.items():
        print(f"类别 {cls}: {ratio:.2%}")

    # Calculate the number of grids after clustering
    def find_clusters(grid):
        visited = np.zeros_like(grid, dtype=bool)
        clusters = []

        def bfs(x, y, label):
            queue = [(x, y)]
            visited[x, y] = True
            cluster_cells = [(x, y)]  # 记录该聚类的所有单元格

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四邻域搜索
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if not visited[nx, ny] and grid[nx, ny] == label:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                            cluster_cells.append((nx, ny))

            return cluster_cells

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 0 and not visited[i, j]:  # 忽略 0（未聚类区域）
                    cluster_size = len(bfs(i, j, grid[i, j]))
                    clusters.append(cluster_size)

        return clusters

    clusters = find_clusters(grid)

    # Calculate the final number of meshes
    num_final_grids = len(clusters)

    # Calculate the average grid size
    avg_grid_area = np.mean(clusters)
    avg_grid_length = np.mean([np.sqrt(size) for size in clusters])

    # Calculate the grid merge ratio
    original_num_grids = num * num
    merge_ratio = (original_num_grids - num_final_grids) / original_num_grids

    # Calculate grid density

    grid_density = num_final_grids / (num * num)



    plt.rcParams['font.sans-serif'] = ['SimHei']  #
    plt.rcParams['axes.unicode_minus'] = False  #


    # Counting the final number of grids at different scales
    final_grid_counts = defaultdict(int)

    # Recording of merged grids
    visited = np.zeros((num, num), dtype=bool)

    # Recursively perform a quadtree merge
    def quadtree(matrix, x, y, size):
        """ Recursively perform quadtree merging and count the number of grids that are eventually merged """
        # global visited

        # If the current grid has already been merged, skip the
        if visited[x:x + size, y:y + size].all():
            return True

        # Fetch the contents of the current size×size grid.
        sub_matrix = matrix[x:x + size, y:y + size]

        # Determine if merging is possible (same value for entire region)
        if np.all(sub_matrix == sub_matrix[0, 0]):
            final_grid_counts[size] += 1
            visited[x:x + size, y:y + size] = True
            return True
        else:

            new_size = size // 2
            if new_size > 0:
                quadtree(matrix, x, y, new_size)
                quadtree(matrix, x + new_size, y, new_size)
                quadtree(matrix, x, y + new_size, new_size)
                quadtree(matrix, x + new_size, y + new_size, new_size)
            return False

    # Quadtree merging on a scale from largest to smallest
    sizes = [128, 64, 32, 16, 8, 4, 2, 1]
    for size in sizes:
        for i in range(0, num, size):
            for j in range(0, num, size):
                if not visited[i, j]:  # Only areas that have not been merged are attempted to be merged
                    quadtree(grid, i, j, size)

    # output
    total_count = sum(final_grid_counts.values())
    for size, count in sorted(final_grid_counts.items(), reverse=True):
        print(f"{size}×{size} 级别的最终单元格数量: {count}")

    print(f"\n最终总单元格数量: {total_count}")
    print(f"聚类后网格数目: {num_final_grids}")
    print(f"聚类后网格平均面积: {avg_grid_area:.2f} 个单元")
    print(f"聚类后网格平均边长: {avg_grid_length:.2f} 单元")
    print(f"网格合并比例: {merge_ratio:.2%}")
    print(f"网格密度: {grid_density:.6f} （单位面积网格数）")





