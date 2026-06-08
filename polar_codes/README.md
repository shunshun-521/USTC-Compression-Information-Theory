# 极化码编译码仿真

纯 Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似（GA）构造
- `encoder.py` — XOR 蝶形编码（与标准因子图一致）
- `decoder_sc.py` / `decoder_scl.py` / `decoder_bp.py` — 译码器
- `simulation.py` / `utils.py` — 仿真与绘图
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本
- `results/` — CSV 与图像输出

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py   # 验证 GA 构造
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

说明：SCL 列表大小 `L≥4` 在 `N=512` 时计算量很大（`L=8` 单帧可达数分钟）。实验二默认 `L∈{2,4}`；CA-SCL 使用 `L=4`。

## 编码器自检

`u=[1,0,1,1]` 时码字为 `[1,1,0,1]`（XOR 蝶形编码，与 Arikan 因子图一致）。
