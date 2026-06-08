"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _generate_matrix(N):
    """生成极化码生成矩阵 G_N = F^{\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码。

    蝶形实现与矩阵乘法 x = u * G_N 等价（G_N = F^{\otimes n}）。
    信道传输顺序为自然索引；SC 译码器与之匹配。
    """
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    v = u.copy()
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                v[i + j] ^= v[i + j + step]
    return v


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = u @ _generate_matrix(4) % 2
    print(f"u={u} -> x={x}, matmul={x_ref}")
    assert np.array_equal(x, x_ref), f"编码器与矩阵乘法不一致: {x} vs {x_ref}"
    print("编码器校验通过")
