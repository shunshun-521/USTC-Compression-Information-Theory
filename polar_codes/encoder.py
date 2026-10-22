"""
极化码编码器
编码：x = u * G_N，利用蝶形递归结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _polar_transform_recursive(u):
    """Arikan 递归极化变换（与 SC 译码因子图一致）"""
    N = len(u)
    if N == 1:
        return u.copy()
    half = N // 2
    left = _polar_transform_recursive(u[:half])
    right = _polar_transform_recursive(u[half:])
    out = np.empty(N, dtype=u.dtype)
    out[:half] = left ^ right
    out[half:] = right
    return out


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    assert N & (N - 1) == 0, "N must be power of 2"
    return _polar_transform_recursive(u).astype(np.int8)


def polar_generator_matrix(N):
    """生成与 polar_encode 一致的 G 矩阵（用于校验）"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[i] = polar_encode(e)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = polar_generator_matrix(4)
    x2 = (u @ G) % 2
    print("matrix multiply:", x2)
