# 极化码编译码仿真（Python）

纯 NumPy/SciPy 实现：GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python3 validate.py
python3 construction.py
```

## 实验

```bash
cd polar_codes
python3 run_exp1.py   # SC：N=256,512,1024
python3 run_exp2.py   # SCL / CA-SCL
python3 run_exp3.py   # BP 与 SC/SCL 对比
```

快速调试（缩短仿真）：

```bash
POLAR_MAX_FRAMES=2000 POLAR_MIN_ERRORS=20 POLAR_SKIP_N1024=1 python3 run_exp1.py
```

## 说明

- 编码器对 `u=[1,0,1,1]` 输出 `x=[1,0,1,1]`（与部分教材手算 `[0,0,1,1]` 的索引约定不同，但编码/译码内部一致）。
- LLR：`2y/σ²`，正号倾向比特 0。
