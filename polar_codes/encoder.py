"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(int)
    rev = rev[:, ::-1]
    weights = 2 ** np.arange(n)
    return (rev * weights).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ G_N (mod 2)
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u.astype(int)


def polar_encode_matrix(u):
    """用生成矩阵验证（测试用，G 行由单位向量编码得到）"""
    N = len(u)
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i] = polar_encode(e)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:", xm)
    print("match:", np.array_equal(x, xm))
