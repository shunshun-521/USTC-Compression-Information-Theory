# 极化码编译码仿真

Python + NumPy/SciPy 实现（无第三方极化码库）。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 verify.py
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

## 说明

- 编码：`polar_encode` 为蝶形变换 + 比特倒序（BRP）。
- SC/SCL：Permuted SCD，信道 LLR 需与编码一致的 BRP 顺序（`align_channel_llr`）。
- 编码器自检：`u=[1,0,1,1]` → `x=[1,0,1,1]`（标准 `u@G_N` 约定）。

结果输出在 `results/`。
