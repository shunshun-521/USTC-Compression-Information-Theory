"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形结构，每层 u[l] ^= u[l + block]
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                u[start + k] ^= u[start + k + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    # 与 G = F^{\\otimes n} 一致
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F) % 2
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("Encoder test passed.")
