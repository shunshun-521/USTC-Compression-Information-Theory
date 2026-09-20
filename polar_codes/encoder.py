"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_U_TRANSFORM_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix_standard(N):
    """标准 Arikan 生成矩阵 G_N = B_N F^{\\otimes n}（行比特倒序）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G[bit_reversal_permutation(N)]


def _gf2_inverse(M):
    n = M.shape[0]
    aug = np.hstack([M.copy(), np.eye(n, dtype=int)])
    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r, col])
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(n):
            if row != col and aug[row, col]:
                aug[row] = (aug[row] + aug[col]) % 2
    return aug[:, n:]


def _build_generator_matrix_custom(N):
    """通过单位基编码得到自定义 G_N"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[i] = polar_encode(u)
    return G


def get_u_domain_transform(N):
    """
    返回 (A, A_inv)，满足 polar_encode(u) = (u @ A) @ G_std。
    SC 译码在 u' = u @ A 域进行，再经 A_inv 映回原 u 域。
    """
    if N in _U_TRANSFORM_CACHE:
        return _U_TRANSFORM_CACHE[N]

    G_std = build_generator_matrix_standard(N)
    G_custom = _build_generator_matrix_custom(N)
    G_std_inv = _gf2_inverse(G_std)
    A = (G_custom @ G_std_inv) % 2
    A_inv = _gf2_inverse(A)
    _U_TRANSFORM_CACHE[N] = (A, A_inv)
    return A, A_inv


def transform_frozen_bits(frozen_bits, info_indices, N):
    """
    将 u 域冻结掩码映射到 u' = u @ A 域（供标准 SC/SCL/BP 使用）。
    frozen_bits: 0=信息位, 1=冻结位
    """
    A, _ = get_u_domain_transform(N)
    info_mask = np.zeros(N, dtype=bool)
    info_mask[info_indices] = True
    depends_on_info = np.any(A[info_mask, :], axis=0)
    frozen_prime = np.ones(N, dtype=int)
    frozen_prime[depends_on_info] = 0
    return frozen_prime


def map_decoded_to_u_domain(u_prime_hat, N):
    """u' 域译码结果映回原 u 域"""
    _, A_inv = get_u_domain_transform(N)
    return (u_prime_hat @ A_inv) % 2


def polar_encode(u):
    """
    极化码编码（含比特倒序置换与列块置换）。
    等价于 x = u @ A @ G_std。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    if N >= 2:
        u[0], u[1] = u[1], u[0]

    n = int(np.log2(N))
    for step in (1 << s for s in range(n)):
        for j in range(0, N, 2 * step):
            for k in range(j, j + step):
                u[k] ^= u[k + step]

    br = bit_reversal_permutation(N)
    u = u[br]

    for i in range(0, N, 4):
        if i + 4 <= N:
            block = u[i:i + 4].copy()
            u[i:i + 4] = block[[1, 2, 0, 3]]

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("encoder test passed")

    A, A_inv = get_u_domain_transform(4)
    G_std = build_generator_matrix_standard(4)
    assert np.array_equal((u @ A @ G_std) % 2, x)
    print("transform consistency passed")
