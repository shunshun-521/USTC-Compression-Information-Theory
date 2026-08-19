"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{\\otimes n} * B_N
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[bit_reverse(i)]"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_butterfly(u):
    """蝶形 XOR 编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = N // 2
    while step > 0:
        for j in range(0, N, 2 * step):
            for i in range(step):
                u[j + i] ^= u[j + i + step]
        step //= 2
    return u.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    与 aff3ct 非系统化编码器相比，末尾额外施加 B_N 置换。
    """
    u = polar_encode_butterfly(u)
    br = bit_reversal_permutation(len(u))
    return u[br].copy()


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xb = polar_encode_butterfly(u)
    print("u:", u, "butterfly:", xb, "with br:", x)
