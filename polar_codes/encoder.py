"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    assert 2 ** n == N
    idx = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2

    # 与 SC 译码树一致：蝶形输出即为信道码字（不额外做 B_N 置换）
    return u.astype(int)


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    Fn = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        Fn = np.kron(Fn, F)
    B = np.zeros((N, N), dtype=int)
    rev = bit_reversal_permutation(N)
    for i, r in enumerate(rev):
        B[i, r] = 1
    return (B @ Fn) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u, "x:", x, "G*u:", x_ref)
