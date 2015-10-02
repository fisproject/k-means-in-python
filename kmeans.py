#coding:utf-8
import numpy as np
import sklearn.decomposition as decop
import matplotlib.pyplot as plt
import sys

# 標準化
def scale(X):
    col = X.shape[1]
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    for i in range(col):
        X[:,i] = (X[:,i] - mu[i]) / sigma[i]

    return X

# 所属クラスタの割り当て : 各piを固定して, eを∂について最小化する
def D(K, N, r):
    for n in range(N):
        i = -1
        min = sys.maxint
        
        for k in range(K):
            tmp = np.linalg.norm(X[n] - mean[k]) ** 2
            if tmp < min:
                i = k
                min = tmp

        for k in range(K):
            if k == i:
                r[n, k] = 1
            else:
                r[n, k] = 0
    return r

# 平均ベクトルの算出 : 各∂iを固定して, eをpについて最小化する
def P(K, N, mean):
    for k in range(K):
        numerator = 0.0
        denominator = 0.0
        for n in range(N):
            numerator += r[n, k] * X[n]
            denominator += r[n, k]
        mean[k] = numerator / denominator

    return mean

# 量子化誤差
def E(X, mean, r):
    e = 0.0
    for n in range(len(X)):
        tmp = 0.0
        for k in range(K):
            tmp += r[n, k] * np.linalg.norm(X[n]-mean[k]) ** 2
        e += tmp

    return e

# 収束判定 : 量子化誤差による評価
def convergence_test(p, np):
    e = p - np

    if e < 0.01:
        return (True, e)
    else:
        return (False, e)

# Plot Cluster
def plot_cluster(X, K, N, r):
    colors = ['g','b','r']
    for n in range(N):
        for k in range(K):
            if r[n, k] == 1:
                c = colors[k]
        plt.scatter(X[n,0], X[n,1], c=c, marker='o')

    plt.show()

# Plot Quantization Error
def plot_error(err):
    plt.plot(err)
    plt.xlabel('Iteration')
    plt.ylabel('Quantization Error')
    plt.show()

if __name__ == "__main__":
    # load data
    data = np.loadtxt('data/wine.data', delimiter=',')
    X = scale(data[:,1:14])

    # number of clusters
    K = 3

    # principal component analysis
    pca = decop.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    N = len(X)
    mean = np.random.rand(K, 2)
    r = np.zeros((N, K))
    r[:, 0] = 1

    # STEP 1: 初期状態の設定
    proto = E(X, mean, r)
    t = 0
    err = []

    while True:
        # STEP 2: 所属クラスタの割り当て
        r = D(K, N, r)

        # STEP 3: 平均ベクトルの算出
        mean = P(K, N, mean)

        # STEP 4: 量子化誤差の算出
        new_proto = E(X, mean, r)
        err.append(new_proto)

        # STEP 5: 収束判定
        res = convergence_test(proto, new_proto)

        print 'Iter : ', t, ' Quantization Error : ', new_proto, ' Diff : ', res[1]
        plot_cluster(X, K, N, r)

        if res[0] is False:
            proto = new_proto
            t += 1
        else:
            break

    plot_error(err)
