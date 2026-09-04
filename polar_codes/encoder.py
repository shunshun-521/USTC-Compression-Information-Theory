"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：从大到小阶段，u[i] ^= u[i + step]（左半与右半合并）
    最后对比特倒序置换得到码字。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    stage_len = N
    for _ in range(n):
        half = stage_len // 2
        for block_start in range(0, N, stage_len):
            for k in range(half):
                idx = block_start + k
                u[idx] ^= u[idx + half]
        stage_len = half
    br = bit_reversal_permutation(N)
    return u[br]


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f'u={u} -> x={x}')
    # 手算验证：蝶形变换 + 比特倒序
    assert np.array_equal(x, [1, 0, 1, 1]), f'编码器错误: {x}'
    print('Encoder test passed.')
