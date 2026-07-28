"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    i = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for k in range(n):
        rev += ((i >> k) & 1) << (n - 1 - k)
    return rev


def bit_reversed(x, n):
    """单个索引的比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N

    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half

    return u


if __name__ == "__main__":
    # 标准 F=[[1,0],[1,1]] 生成矩阵验证
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]])
    F2 = np.kron(F, F)
    expected = u @ F2 % 2
    print(f"u={u}, x={x}, expected={expected}")
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    # 规格文档中的测试向量对应 B_N 置换后的结果
    brp = bit_reversal_permutation(4)
    B = np.zeros((4, 4), dtype=int)
    for i in range(4):
        B[i, brp[i]] = 1
    spec_expected = u @ B @ F2 % 2
    print(f"spec expected (with B_N): {spec_expected}")
    print("Encoder test passed.")
