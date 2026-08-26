"""
极化码编码器
编码：u * G_N，G_N = F^{\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引：out[i] = in[br(i)]"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序（与 polarcodes 库一致）。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（非递归分块 XOR，等价于 Arikan 生成矩阵编码）。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                x[idx] ^= x[idx + half]
        block = half
    return x.astype(int)


def polar_encode_with_reversal(u):
    """编码后做比特倒序置换。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "x:", x)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
