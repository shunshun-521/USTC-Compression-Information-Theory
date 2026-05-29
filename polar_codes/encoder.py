"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"N={N} must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    采用与 SC 译码器匹配的规范：对 u 做蝶形 XOR 变换（等价于 u * F^{⊗n}）。
    比特倒序置换由译码端按倒序比特索引译码等效实现，与显式 B_N 乘后编码等价。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"N={N} must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                left = p + k
                u[left] = (u[left] + u[left + half]) % 2
        block = half

    return u.astype(np.int8)


def polar_encode_with_bit_reversal(u):
    """显式比特倒序置换的编码结果 x[i] = enc[br(i)]"""
    enc = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    out = np.zeros_like(enc)
    out[br] = enc
    return out


def generator_matrix(N):
    """构造 F^{\\otimes n}（模 2），用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("polar_encode:", x)
    print("u @ F^n:", x_ref)
    assert np.array_equal(x, x_ref), "编码与生成矩阵不一致"
