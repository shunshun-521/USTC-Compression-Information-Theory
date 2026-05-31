# LaTeX 实验报告

## 文件

| 文件 | 说明 |
|------|------|
| `report.tex` | 已填入仿真数据的报告（由脚本生成） |
| `generate_report_tex.py` | 从 `results/*.csv` 重新生成 `report.tex` |
| `figures/` | 实验图表 PDF |

## 重新生成

```bash
cd polar_codes/report
python3 generate_report_tex.py
```

## 编译（本地 Windows / TeX Live）

需安装 **TeX Live**（含 `xelatex` 与 `ctex` 宏包）：

```bash
cd polar_codes/report
xelatex report.tex
xelatex report.tex   # 第二遍生成目录
```

或在 TeXstudio / Overleaf 中打开 `report.tex`，编译器选 **XeLaTeX**。

## 待手动填写

封面中的姓名、学号、班级、指导教师（搜索 `\placeholder`）。

## 说明

- 实验一：完整 22 个 SNR 点（步长 0.25 dB）
- 实验二/三：快速模式 7 点（2.0–5.0 dB，步长 0.5 dB），与 `REPORT.md` 一致
- 实验二未含 SCL $L=16$（与仿真设置一致）
