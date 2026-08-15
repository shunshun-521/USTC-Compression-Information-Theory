"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    使用蝶形结构：对每对子块执行 con(a,b) = [a XOR b, b]，
    共 n = log2(N) 层，得到 x = u * F^{⊗ n}（无比特倒序）。
    与 SC/SCL 译码器配套使用。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    m = 1
    for _ in range(n):
        for i in range(0, N, 2 * m):
            a = u[i : i + m]
            b = u[i + m : i + 2 * m]
            u[i : i + m] = (a ^ b) % 2
            u[i + m : i + 2 * m] = b
        m *= 2
    return u


def polar_encode_with_brev(u):
    """
    带比特倒序的编码变体（x = u * G_N，G_N = B_N F^{⊗ n}）。
    用于 BP 译码器等需要标准生成矩阵的场景。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    brp = bit_reversal_permutation(N)
    return u[brp]


if __name__ == "__main__":
    from decoder_sc import sc_decode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    uh = sc_decode(np.where(x == 0, 100.0, -100.0), np.zeros(4, dtype=bool))
    assert np.array_equal(uh, u), f"roundtrip failed: {uh}"
    print("u:", u, "-> x:", x, "-> u_hat:", uh)
    print("Encoder test passed.")
