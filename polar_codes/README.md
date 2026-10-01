# Polar Codes 仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、蝶形编码、SC/SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python3 validate.py
```

## 实验

完整仿真（耗时较长）：

```bash
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟（`POLAR_FAST_SIM=1` 降低帧数与 SNR 点数）：

```bash
POLAR_FAST_SIM=1 python3 run_exp1.py
```

## 说明

- SC 译码核心采用 [mcba1n/polar-codes](https://github.com/mcba1n/polar-codes)（MIT）因子图更新，见 `scd_vendor/`。
- 编码为蝶形 XOR（与上述库一致），**不对码字做比特倒序**。
- BP 译码为教学用因子图实现，高码长下性能可能与文献有差距，可与 SC/SCL 曲线对照。
