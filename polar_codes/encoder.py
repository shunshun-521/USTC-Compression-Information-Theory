"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reverse_index(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * F^⊗n（蝶形 XOR，与标准极化码库一致）。
    接收端 LLR 按自然顺序输入译码器。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block_len = N
    for _ in range(n):
        half = block_len // 2
        for base in range(0, N, block_len):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block_len = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # 手算验证：F^{\otimes 2} @ u = [1,1,0,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
