r"""
极化码编码器
编码：x = u * F^{\otimes n}（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    perm = np.zeros(N, dtype=int)
    for i in range(N):
        rev = 0
        for b in range(n):
            if (i >> b) & 1:
                rev |= 1 << (n - 1 - b)
        perm[i] = rev
    return perm


def polar_encode(u):
    """
    极化码编码（蝶形，等价于 u @ F^{\\otimes n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block_len = N
    while block_len > 1:
        half = block_len // 2
        for block_start in range(0, N, block_len):
            for k in range(half):
                idx = block_start + k
                u[idx] ^= u[idx + half]
        block_len = half
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("encode:", x)
    print("matrix:", x_ref)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
