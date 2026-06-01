# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py   # 验证构造
python run_exp1.py       # SC 仿真
python run_exp2.py       # SCL / CA-SCL
python run_exp3.py       # BP 对比
```

结果输出至 `results/`（CSV、PNG/PDF、frozen_sets.txt）。

## 说明

- 编码器默认 `polar_encode(u)` 为蝶形 `F^{\otimes n}`（与分层 SC 译码器配套）；可选 `apply_bit_reversal=True` 得到 `G_N = B_N F^{\otimes n}`。
- SC 译码采用对数域 / min-sum 分层更新；SCL 含 Lazy Copy；BP 含 min-sum 与早停。
