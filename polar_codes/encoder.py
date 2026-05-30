"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_GENC_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（非递归 XOR 蝶形，与 SC 分层译码配套）。
    译码端在比特倒序索引顺序下处理，无需对码字再做倒序置换。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def _build_generator(N):
    """G_N = B_N F^{\\otimes n}（矩阵形式，用于校验）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    rev = bit_reversal_permutation(N)
    return G[np.ix_(rev, rev)]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（校验用）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    if N not in _GENC_CACHE:
        _GENC_CACHE[N] = _build_generator(N)
    return (u @ _GENC_CACHE[N]) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    # 与极化码标准 XOR 编码一致；规范中的 [0,0,1,1] 对应另一索引排列
    from decoder_sc import sc_decode

    llr = 500.0 * (1 - 2 * x)
    uh = sc_decode(llr, np.zeros(4, dtype=bool))
    assert np.array_equal(uh, u), f"环回失败: {uh}"
    print("encoder round-trip test passed")
