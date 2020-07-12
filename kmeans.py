# -*- coding: utf-8 -*-

import sys
from typing import Tuple, List

import numpy as np
import sklearn.decomposition as decop

import matplotlib

matplotlib.use("TkAgg")  # for macOS
import matplotlib.pyplot as plt


def scale(data: np.ndarray) -> np.ndarray:
    col = data.shape[1]
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    for i in range(col):
        data[:, i] = (data[:, i] - mu[i]) / sigma[i]
    return data


class Kmeans(object):
    def __init__(self, data: np.ndarray, n_clusters: int):
        self.X = data
        self.K = n_clusters
        self.mean = np.random.rand(self.K, 2)
        self.r = np.zeros((len(self.X), self.K))
        self.r[:, 0] = 1

    def clustering(self) -> np.ndarray:
        """
        所属クラスタの割り当て: 各 pi を固定して, e を ∂ について最小化
        """
        for n in range(len(self.X)):
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

    def mean_vec(self) -> np.ndarray:
        """
        平均ベクトル: 各 ∂i を固定して, e を p について最小化
        """
        for k in range(self.K):
            numerator = 0.0
            denominator = 0.0
            for n in range(len(self.X)):
                numerator += self.r[n, k] * self.X[n]
                denominator += self.r[n, k]
            self.mean[k] = numerator / denominator
        return self.mean

    def error(self) -> float:
        """
        量子化誤差
        """
        err = 0.0
        for n in range(len(self.X)):
            err += sum(
                [
                    self.r[n, k] * np.linalg.norm(self.X[n] - self.mean[k]) ** 2
                    for k in range(self.K)
                ]
            )
        return err

    def convergence(self, p: float, np: float, th: float) -> Tuple[bool, float]:
        """
        収束判定: 量子化誤差の変動で収束を判定
        """
        err = p - np
        if err < th:
            return (True, err)
        else:
            return (False, err)

    def plot_cluster(self) -> None:
        """
        各クラスタを図示 (2次元)
        """
        colors = ["g", "b", "r"]
        for n in range(len(self.X)):
            c = [colors[k] for k in range(self.K) if self.r[n, k] == 1]
            plt.scatter(self.X[n, 0], self.X[n, 1], c=c, marker="o")
        plt.show()

    def plot_errors(self, errors: List[float]) -> None:
        """
        量子化誤差の推移を図示
        """
        plt.plot(errors)
        plt.xlabel("Iteration")
        plt.ylabel("Quantization Error")
        plt.show()


def main():
    data = np.loadtxt("data/wine.data", delimiter=",")
    X = scale(data[:, 1:14])

    # principal component analysis (PCA)
    pca = decop.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    km = Kmeans(data=X, n_clusters=3)

    # STEP 1: 初期状態の設定
    proto = km.error()
    iteration = 0
    errors = []

    while True:
        # STEP 2: 所属クラスタの割り当て
        km.clustering()

        # STEP 3: 平均ベクトルの計算
        km.mean_vec()

        # STEP 4: 量子化誤差の計算
        new_proto = km.error()
        errors.append(new_proto)

        # STEP 5: 収束判定
        is_convergence, diff = km.convergence(proto, new_proto, th=0.01)
        print(f"iter: {iteration}, quantization error: {new_proto}, error diff: {diff}")

        km.plot_cluster()

        if is_convergence:
            break
        else:
            proto = new_proto  # 量子化誤差の更新
            iteration += 1

    km.plot_errors(errors)


if __name__ == "__main__":
    main()
