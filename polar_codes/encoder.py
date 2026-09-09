"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(x):
    r"""蝶形 XOR 编码：x = u * F^{\otimes n}"""
    N = len(x)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
        step *= 2
    return x


def polar_encode(u):
    r"""
    极化码编码：x = u * F^{\otimes n}（蝶形 XOR）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=int).copy()
    return _butterfly_encode(x)


def polar_encode_with_bit_reversal(u):
    r"""编码并施加 B_N 比特倒序置换：x = u * B_N * F^{\otimes n}"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


def polar_encode_no_reversal(u):
    """编码但不进行输出比特倒序（用于 BP 早停重编码校验）"""
    x = np.array(u, dtype=int).copy()
    return _butterfly_encode(x)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
