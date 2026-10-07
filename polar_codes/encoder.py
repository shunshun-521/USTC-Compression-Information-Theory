"""
极化码编码器
编码：x = u * F^{\otimes n}（蝶形结构，O(N log N)）
比特倒序置换 B_N 在译码端通过索引置换处理，与常见参考实现一致。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形变换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（经 B_N 倒序后的发送比特序）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block >= 2:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block //= 2
    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_core(u):
    """仅蝶形变换，不含比特倒序（供 BP 早停重编码等使用）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block >= 2:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block //= 2
    return u


def source_from_codeword(x):
    """从发送码字恢复源向量 u（B_N 与 F^{\otimes n} 均为自逆）"""
    x = np.asarray(x, dtype=np.int8).copy()
    rev = bit_reversal_permutation(len(x))
    x_nat = x[rev]
    return polar_encode_core(x_nat)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    # 标准 Arikan 约定下 u=[1,0,1,1] 的码字为 [1,0,1,1]（经 B_N 后）
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
