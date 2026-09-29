# Polar Codes (`polar_codes/`)

Python 实现极化码 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真（无第三方极化码库）。

## 快速校验

```bash
cd polar_codes
pip install -r requirements.txt
python3 validate.py
```

## 实验脚本

```bash
python3 run_exp1.py   # SC
python3 run_exp2.py   # SCL / CA-SCL
python3 run_exp3.py   # BP 对比
```

环境变量（加速/裁剪）：

- `POLAR_MAX_FRAMES`, `POLAR_MIN_ERRORS`
- `POLAR_N_LIST`（如 `256,512`）
- `POLAR_EB_START`, `POLAR_EB_STOP`, `POLAR_EB_STEP`
- `POLAR_SKIP_SIM=1`：若已有 CSV 则只重绘图

## 实现要点

- 编码：蝶形 XOR + 比特倒序置换 `polar_encode`
- SC/SCL：Permuted SCD（Vangala 2014），信道 LLR 需 `llr[bit_reversal_permutation(N)]` 再译码
- LLR：BPSK `0→+1, 1→-1`，`LLR = 2y/σ²`
