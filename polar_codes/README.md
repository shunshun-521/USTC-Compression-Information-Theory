# 极化码编译码仿真（Python）

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python3 validate.py
```

## 实验

```bash
python3 run_exp1.py   # SC
python3 run_exp2.py   # SCL / CA-SCL
python3 run_exp3.py   # BP 对比
```

快速冒烟（缩短 Eb/N0 网格与帧数）：

```bash
POLAR_QUICK=1 POLAR_MAX_FRAMES=2000 POLAR_MIN_ERRORS=20 POLAR_SKIP_N1024=1 python3 run_exp1.py
```

## 说明

- SC/SCL 采用 Permuted SCD（`L[:,0]` 为信道 LLR，按 bit-reversed 相位译码），编码为蝶形 XOR（与 `u @ F^{\otimes n}` 矩阵形式一致）。
- 环境变量：`POLAR_MAX_FRAMES`、`POLAR_MIN_ERRORS`、`POLAR_SKIP_N1024`、`POLAR_N256_ONLY`、`POLAR_QUICK`。
