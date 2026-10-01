# 极化码编译码仿真

纯 NumPy/SciPy 实现的极化码（GA 构造、SC/SCL/CA-SCL/BP）蒙特卡洛仿真。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 validate.py
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

可通过环境变量缩短仿真时间，例如：

```bash
export POLAR_MAX_FRAMES=5000
export POLAR_MIN_ERRORS=100
export POLAR_SKIP_N1024=1
```

## 说明

- 编码器采用蝶形结构并在输出端做比特倒序置换；SC/SCL 译码器对信道 LLR 做一致的倒序重排（Permuted SCD）。
- 编码自检：`u=[1,0,1,1]` 时 `x=[1,0,1,1]`（对应 \(G_N=B_N F^{\otimes n}\)）。
