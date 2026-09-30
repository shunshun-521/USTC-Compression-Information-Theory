# 极化码编译码仿真

Python/NumPy 实现的极化码 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真。

## 安装

```bash
pip install -r requirements.txt
```

## 运行实验

```bash
cd polar_codes
python run_exp1.py   # SC
python run_exp2.py   # SCL / CA-SCL
python run_exp3.py   # SC vs SCL vs BP
```

快速冒烟（环境变量）：

```bash
POLAR_FAST=1 POLAR_MAX_FRAMES=500 POLAR_MIN_ERRORS=10 python run_exp1.py
```

## 说明

- SC 译码核心为状态机实现（`sc_decoder_impl.py`，与蝶形编码 `F^{⊗n}` 一致）。
- SCL 当前列表译码与 SC 共用内核；完整多路径 SCL 可在此基础上扩展。
- BP 译码器为因子图 min-sum 实现，参数需与码长/信噪比联合调优。

结果输出在 `results/` 目录。
