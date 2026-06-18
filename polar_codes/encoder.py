"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    result = np.zeros(N, dtype=int)
    for i in range(N):
        rev = 0
        x = i
        for _ in range(n):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        result[i] = rev
    return result


def polar_encode(u):
    """
    极化码编码（标准生成矩阵，蝶形递归结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：递归蝶形结构，复杂度 O(N log N)。
    """
    u = np.asarray(u, dtype=np.int8)

    def _encode(bits):
        n = len(bits)
        if n == 1:
            return bits.copy()
        half = n // 2
        left = _encode(np.bitwise_xor(bits[:half], bits[half:]))
        right = _encode(bits[half:].copy())
        out = np.empty(n, dtype=np.int8)
        out[:half] = left
        out[half:] = right
        return out

    return _encode(u)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
