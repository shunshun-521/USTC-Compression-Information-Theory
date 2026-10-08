# 极化码编译码仿真

纯 NumPy/SciPy/Matplotlib 实现（无第三方极化码库）。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

快速冒烟（减少帧数与 SNR 点）：`POLAR_QUICK=1 python run_exp1.py`

## 说明

- 编码采用 Kronecker 蝶形 XOR（自然比特序，与 SC/SCL 一致）。
- SC/SCL 使用精确 log 域 `f` 运算；`list_size=1` 时 SCL 等价于 SC。
- 单元测试见各 `run_exp*.py` 开头。
