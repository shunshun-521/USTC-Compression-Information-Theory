"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode_core(u):
    """蝶形编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=int).copy()
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


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * B_N * F^{\\otimes n}
    """
    u_work = polar_encode_core(u)
    rev = bit_reversal_permutation(len(u_work))
    return u_work[rev]


def polar_encode_no_reversal(u):
    """极化码编码（不含比特倒序），x = u * F^{\\otimes n}。"""
    return polar_encode_core(u)


if __name__ == "__main__":
    # 标准 B@F 编码校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")

    # 与生成矩阵对照
    N = 4
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    rev = bit_reversal_permutation(N)
    for i in range(N):
        B[rev[i], i] = 1
    G_full = (B @ G) % 2
    x_mat = (u @ G_full) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("编码器矩阵校验通过")
