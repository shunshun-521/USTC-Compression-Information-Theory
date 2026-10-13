"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int
    )


def bit_reversed(i, n):
    """单索引比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，与标准 SCD 因子图一致）。
    不在码字末尾额外做比特倒序；信道输出顺序与 u 经 F^{⊗n} 变换后的顺序一致。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_matrix(N):
    """生成矩阵 G_N（用于校验），GF(2)"""
    I = np.eye(N, dtype=np.int8)
    cols = np.stack([polar_encode(I[:, i]) for i in range(N)], axis=1)
    return cols.T


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    print("encode u=", u, "-> x=", x, "u@G=", (u @ G) % 2)
