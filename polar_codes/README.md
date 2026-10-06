# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python validate.py
```

## 实验

完整仿真（耗时较长）：

```bash
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

快速冒烟（环境变量 `POLAR_QUICK=1`）：

```bash
POLAR_QUICK=1 python run_exp1.py
```

## 说明

- 信道 LLR 在送入 SC/SCL/BP 前需做比特倒序置换：`llr[bit_reversal_permutation(N)]`（`simulation.py` 已处理）。
- SC 递归译码的 `g` 运算使用左子树蝶形部分和（见 `decoder_sc._partial_sums`）。
