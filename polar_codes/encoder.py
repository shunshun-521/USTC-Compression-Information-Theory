"""
极化码编码器
编码：u 经蝶形 XOR（与 F^{\otimes n} 等价），可选比特倒序置换
"""
import numpy as np


def bit_reversal_permutation(N):
    """gather 置换：out[i] = in[perm[i]]"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """与 F^{\otimes n} 对应的非递归 XOR 编码（参考 polar_encode 实现）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u, bit_reverse=True):
    """
    极化码编码。
    bit_reverse=True 时输出 x[i]=v[bit_reverse(i)]，与常见 G_N=B_N F^{\otimes n} 描述一致。
    """
    v = _butterfly_encode(u)
    if bit_reverse:
        return v[bit_reversal_permutation(len(u))]
    return v


def polar_generator_matrix(N, with_bit_reverse=True):
    """G_N（用于校验）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    g = np.array([[1]], dtype=int)
    for _ in range(n):
        g = np.kron(g, F) % 2
    if with_bit_reverse:
        br = bit_reversal_permutation(N)
        b = np.zeros((N, N), dtype=int)
        for i, j in enumerate(br):
            b[i, j] = 1
        return (b @ g) % 2
    return g % 2


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x_br = polar_encode(u, bit_reverse=True)
    x_nb = polar_encode(u, bit_reverse=False)
    G = polar_generator_matrix(4, with_bit_reverse=True)
    Gf = polar_generator_matrix(4, with_bit_reverse=False)
    assert np.array_equal(x_br, (u @ G) % 2)
    assert np.array_equal(x_nb, (u @ Gf) % 2)
    return True


if __name__ == "__main__":
    validate_encoder()
    print("encoder OK")
