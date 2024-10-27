import numpy as np
# from math import radians, cos, sin, sqrt, atan2

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        n_samples, _ = X.shape
        # 随机选择初始中心
        random_state = np.random.RandomState(self.random_state)
        self.centers = X[random_state.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # 计算每个样本到中心的距离，并分配到最近的中心
            distances = self._great_circle_distance(X, self.centers)
            labels = np.argmin(distances, axis=0)
            # 更新中心
            # new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            new_centers = []
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(axis=0))
                else:
                    # Handle the empty cluster, e.g., by selecting a random point or using a previous center
                    new_centers.append(X[np.random.choice(X.shape[0])])

            new_centers = np.array(new_centers)
            # 检查中心是否收敛
            if np.all(self.centers == new_centers):
                break
            self.centers = new_centers
    def predict(self, X):
        distances = self._great_circle_distance(X, self.centers)
        return np.argmin(distances, axis=0)
    def _great_circle_distance(self, X, centers):
        # 将经纬度转换为弧度
        X = np.radians(X)
        centers = np.radians(centers)
        # 计算大圆距离
        dlat = centers[:, 1][:, np.newaxis] - X[:, 1]
        dlon = centers[:, 0][:, np.newaxis] - X[:, 0]
        a = np.sin(dlat / 2)**2 + np.cos(X[:, 1]) * np.cos(centers[:, 1][:, np.newaxis]) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = 6371 * c  # 地球平均半径约为6371公里
        return d
    

class KMeansNew:
    def __init__(self, n_clusters=3, max_iter=300, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        n_samples, _ = X.shape
        random_state = np.random.RandomState(self.random_state)
        # 随机选择初始中心
        self.centers = X[random_state.choice(n_samples, self.n_clusters, replace=False), :2]  # 只选择经纬度作为中心
        # 初始化每个聚类的时间总和
        cluster_time_sums = np.zeros(self.n_clusters)
        for _ in range(self.max_iter):
            distances = self._great_circle_distance(X[:, :2], self.centers)
            labels = np.argmin(distances, axis=0)
            new_centers = []
            new_cluster_time_sums = np.zeros(self.n_clusters)
            # 计算每个聚类的新中心和时间总和
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points[:, :2].mean(axis=0))
                    new_cluster_time_sums[i] = cluster_points[:, 2].sum()
                else:
                    # 处理空聚类
                    selected = np.random.choice(n_samples)
                    new_centers.append(X[selected, :2])
                    new_cluster_time_sums[i] = X[selected, 2]

            new_centers = np.array(new_centers)
            # 如果时间总和更均匀或中心几乎不变，可以终止
            if np.allclose(cluster_time_sums, new_cluster_time_sums, atol=1800) and np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers
            cluster_time_sums = new_cluster_time_sums

    def predict(self, X):
        distances = self._great_circle_distance(X[:, :2], self.centers)
        return np.argmin(distances, axis=0)

    def _great_circle_distance(self, X, centers):
        X = np.radians(X)
        centers = np.radians(centers)
        dlat = centers[:, 1][:, np.newaxis] - X[:, 1]
        dlon = centers[:, 0][:, np.newaxis] - X[:, 0]
        a = np.sin(dlat / 2)**2 + np.cos(X[:, 1]) * np.cos(centers[:, 1][:, np.newaxis]) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = 6371 * c  # 地球平均半径约为6371公里
        return d

# # 使用示例
# if __name__ == "__main__":
#     from sklearn.datasets import make_blobs
#     from matplotlib import pyplot as plt

#     # 生成示例数据
#     X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#     print(X.shape)

#     # 使用KMeans进行聚类
#     kmeans = KMeans(n_clusters=4)
#     kmeans.fit(X)

#     # 可视化结果
#     plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap='viridis')
#     plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=300, c='red', marker='X')
#     plt.title('K-Means Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.show()
