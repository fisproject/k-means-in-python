# -*- coding: utf-8 -*-

import sys
from typing import Tuple, List

import numpy as np
import sklearn.decomposition as decop

import matplotlib

matplotlib.use("TkAgg")  # for macOS
import matplotlib.pyplot as plt


# 標準化
def scale(data: np.ndarray) -> np.ndarray:
    col = data.shape[1]
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    for i in range(col):
        data[:, i] = (data[:, i] - mu[i]) / sigma[i]
    return data


class Kmeans(object):
    def __init__(self, data: np.ndarray, clusters: int):
        self.X = data
        self.N = len(self.X)
        self.K = clusters
        self.mean = np.random.rand(self.K, 2)
        self.r = np.zeros((self.N, self.K))
        self.r[:, 0] = 1

    # 所属クラスタの割り当て : 各piを固定して, eを∂について最小化する
    def clustering(self) -> np.ndarray:
        for n in range(self.N):
            i = -1
            min_ = sys.maxsize

            for k in range(self.K):
                tmp = np.linalg.norm(self.X[n] - self.mean[k]) ** 2
                if tmp < min_:
                    i = k
                    min_ = tmp

            for k in range(self.K):
                if k == i:
                    self.r[n, k] = 1
                else:
                    self.r[n, k] = 0
        return self.r

    # 平均ベクトルの算出 : 各∂iを固定して, eをpについて最小化する
    def mean_vec(self) -> np.ndarray:
        for k in range(self.K):
            numerator = 0.0
            denominator = 0.0
            for n in range(self.N):
                numerator += self.r[n, k] * self.X[n]
                denominator += self.r[n, k]
            self.mean[k] = numerator / denominator
        return self.mean

    # 量子化誤差
    def error(self) -> float:
        err = 0.0
        for n in range(self.N):
            err += sum(
                [
                    self.r[n, k] * np.linalg.norm(self.X[n] - self.mean[k]) ** 2
                    for k in range(self.K)
                ]
            )
        return err

    # 収束判定 : 量子化誤差の変動で収束を判定する
    def check_convergence(self, p: float, np: float) -> Tuple[bool, float]:
        err = p - np
        if err < 0.01:
            return (True, err)
        else:
            return (False, err)

    # Plot Cluster
    def plot_cluster(self) -> None:
        colors = ["g", "b", "r"]
        for n in range(self.N):
            c = [colors[k] for k in range(self.K) if self.r[n, k] == 1]
            plt.scatter(self.X[n, 0], self.X[n, 1], c=c, marker="o")
        plt.show()

    # Plot Quantization Error
    def plot_error(self, err: List[float]):
        plt.plot(err)
        plt.xlabel("Iteration")
        plt.ylabel("Quantization Error")
        plt.show()


def main():
    # load data
    data = np.loadtxt("data/wine.data", delimiter=",")
    X = scale(data[:, 1:14])

    # principal component analysis
    pca = decop.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    # number of clusters: 3
    km = Kmeans(data=X, clusters=3)

    # STEP 1: 初期状態の設定
    proto = km.error()
    iteration = 0
    errs = []

    while True:
        # STEP 2: 所属クラスタの割り当て
        km.clustering()

        # STEP 3: 平均ベクトルの計算
        km.mean_vec()

        # STEP 4: 量子化誤差の計算
        new_proto = km.error()
        errs.append(new_proto)

        # STEP 5: 収束判定
        res = km.check_convergence(proto, new_proto)
        print(f"iter: {iteration}, quantization Error: {new_proto}, diff: {res[1]}")

        km.plot_cluster()

        if res[0] is False:
            # 量子化誤差の保持
            proto = new_proto
            iteration += 1
        else:
            break

    km.plot_error(errs)


if __name__ == "__main__":
    main()
