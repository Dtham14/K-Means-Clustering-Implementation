import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str = 'random', max_iter=300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None  # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        # print(self.centroids)

        while iteration < self.max_iter:
            # your code

            for i, row in enumerate(X):
                clustering[i] = self.closest_centroid(row)

            old_centroids = self.centroids
            self.centroids = self.update_centroids(clustering, X)

            # np.savetxt("out" + str(iteration) + ".csv",
            #            clustering, delimiter=",")

            if self._is_converged(old_centroids, self.centroids):
                break

            # print(iteration)
            # print(clustering)
            iteration += 1

        return clustering

    def _is_converged(self, centroids_old, centroids):
        # helper to determine if we have the same labels to help stop printing the iterations
        distances = [self.euclidean_distance(
            centroids_old[i], centroids[i]) for i in range(self.n_clusters)]
        return sum(distances) == 0

    def closest_centroid(self, row: np.ndarray):
        # helper to determine the closest centroid in each iteration of the kmeans
        euclid_dist = [self.euclidean_distance(
            row, point) for point in self.centroids]
        closest_centroid = np.argmin(euclid_dist)
        return closest_centroid

    def next_closest_centroid(self, row: np.ndarray):
        # helper to determine the second closest centroid for b in silhouette
        euclid_dist = [self.euclidean_distance(
            row, point) for point in self.centroids]
        sorted_centroid = np.sort(euclid_dist)
        return sorted_centroid[1]

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        # find mean of clustering array to update each iteration
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for cluster in np.unique(clustering):
            clusters = []
            for index, _ in enumerate(clustering):
                if (clustering[index] == cluster):
                    clusters.append(X[index])
            cluster_mean = np.mean(clusters, axis=0)
            centroids[int(cluster)] = cluster_mean
        return centroids

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            np.random.seed(42)
            self.centroids = []
            m = np.shape(X)[0]

            for _ in range(self.n_clusters):
                r = np.random.randint(0, m-1)
                self.centroids.append(X[r])

            return np.array(self.centroids)

        elif self.init == 'kmeans++':
            # your code
            self.centroids = [X[0]]

            for _ in range(1, self.n_clusters):
                dist_sq = np.array([min([np.inner(c-x, c-x)
                                   for c in self.centroids]) for x in X])
                probs = dist_sq/dist_sq.sum()
                cumulative_probs = probs.cumsum()
                # print(cumulative_probs)
                r = np.random.rand()

                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        i = j
                        break

                self.centroids.append(X[i])

            return np.array(self.centroids)
        else:
            raise ValueError(
                'Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1: np.ndarray, X2: np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        dist = np.sqrt(sum(np.power(X2 - X1, 2)))
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        silarray = []

        for i, row in enumerate(X):
            a = self.euclidean_distance(
                row, self.centroids[int(clustering[i])])
            b = self.next_closest_centroid(row)
            silScore = (b-a)/np.maximum(a, b)
            silarray.append(silScore)
        return np.mean(silarray)
