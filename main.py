import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

# python3 main.py --n-clusters 10 --data 'Heart-counts.csv'


def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')

    a = parser.parse_args()
    return (a.n_clusters, a.data)


def read_data(data_path):
    return anndata.read_csv(data_path)


def preprocess_data(adata: anndata.AnnData, scale: bool = True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)


def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    # Your code

    # test = KMeans(n_clusters=n_classifiers, init='random', max_iter=300)
    # result = test.fit(X)
    # print(result)
    # score = test.silhouette(result, X)
    # print(score)

    # random_silhouette_coefficient = []
    # kmeans_silhouette_coefficient = []

    # Random

    # random2 = KMeans(n_clusters=2, init='random', max_iter=300)
    # rres2 = random2.fit(X)
    # rscore2 = random2.silhouette(rres2, X)
    # random_silhouette_coefficient.append(rscore2)

    random3 = KMeans(n_clusters=3, init='random', max_iter=300)
    rres3 = random3.fit(X)
    # rscore3 = random3.silhouette(rres3, X)
    # random_silhouette_coefficient.append(rscore3)

    # random4 = KMeans(n_clusters=4, init='random', max_iter=300)
    # rres4 = random4.fit(X)
    # rscore4 = random4.silhouette(rres4, X)
    # random_silhouette_coefficient.append(rscore4)

    # random5 = KMeans(n_clusters=5, init='random', max_iter=300)
    # rres5 = random5.fit(X)
    # rscore5 = random5.silhouette(rres5, X)
    # random_silhouette_coefficient.append(rscore5)

    # random6 = KMeans(n_clusters=6, init='random', max_iter=300)
    # rres6 = random6.fit(X)
    # rscore6 = random6.silhouette(rres6, X)
    # random_silhouette_coefficient.append(rscore6)

    # random7 = KMeans(n_clusters=7, init='random', max_iter=300)
    # rres7 = random7.fit(X)
    # rscore7 = random7.silhouette(rres7, X)
    # random_silhouette_coefficient.append(rscore7)

    # random8 = KMeans(n_clusters=8, init='random', max_iter=300)
    # rres8 = random8.fit(X)
    # rscore8 = random8.silhouette(rres8, X)
    # random_silhouette_coefficient.append(rscore8)

    # random9 = KMeans(n_clusters=9, init='random', max_iter=300)
    # rres9 = random9.fit(X)
    # rscore9 = random9.silhouette(rres9, X)
    # random_silhouette_coefficient.append(rscore9)

    # # Kmeans

    # kmeans2 = KMeans(n_clusters=2, init='kmeans++', max_iter=300)
    # kres2 = kmeans2.fit(X)
    # kscore2 = kmeans2.silhouette(kres2, X)
    # kmeans_silhouette_coefficient.append(kscore2)

    # kmeans3 = KMeans(n_clusters=3, init='kmeans++', max_iter=300)
    # kres3 = kmeans3.fit(X)
    # kscore3 = kmeans3.silhouette(kres3, X)
    # kmeans_silhouette_coefficient.append(kscore3)

    # kmeans4 = KMeans(n_clusters=4, init='kmeans++', max_iter=300)
    # kres4 = kmeans4.fit(X)
    # kscore4 = kmeans4.silhouette(kres4, X)
    # kmeans_silhouette_coefficient.append(kscore4)

    # kmeans5 = KMeans(n_clusters=5, init='kmeans++', max_iter=300)
    # kres5 = kmeans5.fit(X)
    # kscore5 = kmeans5.silhouette(kres5, X)
    # kmeans_silhouette_coefficient.append(kscore5)

    # kmeans6 = KMeans(n_clusters=6, init='kmeans++', max_iter=300)
    # kres6 = kmeans6.fit(X)
    # kscore6 = kmeans6.silhouette(kres6, X)
    # kmeans_silhouette_coefficient.append(kscore6)

    # kmeans7 = KMeans(n_clusters=7, init='kmeans++', max_iter=300)
    # kres7 = kmeans7.fit(X)
    # kscore7 = kmeans7.silhouette(kres7, X)
    # kmeans_silhouette_coefficient.append(kscore7)

    # kmeans8 = KMeans(n_clusters=8, init='kmeans++', max_iter=300)
    # kres8 = kmeans8.fit(X)
    # kscore8 = kmeans8.silhouette(kres8, X)
    # kmeans_silhouette_coefficient.append(kscore8)

    # kmeans9 = KMeans(n_clusters=9, init='kmeans++', max_iter=300)
    # kres9 = kmeans9.fit(X)
    # kscore9 = kmeans9.silhouette(kres9, X)
    # kmeans_silhouette_coefficient.append(kscore9)

    # print("Random")
    # print(random_silhouette_coefficient)

    # print("Kmeans")
    # print(kmeans_silhouette_coefficient)

    # plt.figure(figsize=(12, 12))
    # plt.subplot(211)
    # plt.title("Plot of k from 2-9 for the Silhouette Coefficient for Random")
    # plt.plot([2, 3, 4, 5, 6, 7, 8, 9], random_silhouette_coefficient, 'ro')

    # plt.subplot(212)
    # plt.title("Plot of k from 2-9 for the Silhouette Coefficient for Kmeans++")
    # plt.plot([2, 3, 4, 5, 6, 7, 8, 9], kmeans_silhouette_coefficient, 'bo')
    # plt.savefig('silhouette_coefficient.png')

    X = PCA(heart.X, 2)
    visualize_cluster(X[:, 0], X[:, 1], rres3)


def visualize_cluster(x, y, clustering):
    # Your code
    import matplotlib.cm as cm

    vals = np.linspace(0, 1, 256)
    np.random.shuffle(vals)
    cluster_labels = np.unique(clustering)

    # https://www.programcreek.com/python/example/90948/matplotlib.cm.rainbow
    colors = cm.rainbow(np.linspace(0, 1, len(cluster_labels)))

    plt.figure(2)
    # print(labels)
    plt.title("Scatter Plot of clusters of best k = 3 [Random Initialization]")
    for i, c in zip(cluster_labels, colors):
        plt.scatter(x[clustering == i], y[clustering == i], s=20, color=c)
    plt.savefig('clustering.png')
    return


if __name__ == '__main__':
    main()
