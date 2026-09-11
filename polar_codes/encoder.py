"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_PERM_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _polar_encode_core(u):
    """蝶形编码（不含比特倒序）。"""
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 2 ** stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] = (u[j] ^ u[j + step]) & 1
    return u


def _polar_encode_decoder_order(u):
    """与标准 SC 因子图一致的编码顺序（大块蝶形，无比特倒序）。"""
    u = np.array(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] = (u[p + k] ^ u[p + k + half]) & 1
        block = half
    return u


def compute_channel_permutation(N):
    """
    计算用户编码器输出到 SC 译码树信道顺序的置换。
    x_user[i] = x_decoder[perm[i]]
    """
    if N in _PERM_CACHE:
        return _PERM_CACHE[N]

    basis_user = np.array(
        [_polar_encode_core((np.arange(N) == j).astype(int))[bit_reversal_permutation(N)]
         for j in range(N)]
    )
    basis_dec = np.array(
        [_polar_encode_decoder_order((np.arange(N) == j).astype(int)) for j in range(N)]
    )

    perm = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if np.array_equal(basis_user[i], basis_dec[j]):
                perm[i] = j
                break

    _PERM_CACHE[N] = perm
    return perm


def channel_llr_to_decoder(llr_ch, N):
    """将信道 LLR 映射到 SC/SCL/BP 译码器使用的顺序。"""
    perm = compute_channel_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[perm] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    x = _polar_encode_core(u)
    return x[bit_reversal_permutation(N)]


def polar_encode_matrix(N):
    """构造生成矩阵 G_N（用于验证）"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[i] = polar_encode(u)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
