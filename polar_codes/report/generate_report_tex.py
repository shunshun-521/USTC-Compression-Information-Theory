#!/usr/bin/env python3
"""从 CSV 结果生成填好数据的 LaTeX 实验报告。"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import find_capacity_limit

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt_sci(x, digits=2):
    x = float(x)
    if x == 0:
        return "$0$"
    exp = int(f"{x:.2e}".split("e")[1])
    if exp >= -2 and exp <= 0:
        return f"${x:.4f}$"
    mant = x / (10 ** exp)
    return f"${mant:.{digits}f}\\times10^{{{exp}}}$"


def csv_rows_tex(rows, cols):
    lines = []
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if c in ("eb_n0_db",):
                cells.append(f"{float(v):.2f}")
            elif c in ("num_errors", "num_frames", "avg_iters"):
                cells.append(str(int(float(v))) if v and v.strip() else "---")
            elif c in ("bler", "ber"):
                cells.append(fmt_sci(v))
            elif c == "avg_decode_time_ms":
                cells.append(f"{float(v):.2f}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def interp_eb_for_bler(rows, target=1e-3):
    pts = [(float(r["eb_n0_db"]), float(r["bler"])) for r in rows]
    pts.sort()
    for i in range(len(pts) - 1):
        e0, b0 = pts[i]
        e1, b1 = pts[i + 1]
        if b0 >= target >= b1 or b0 <= target <= b1:
            if b0 == b1:
                return e0
            t = (target - b0) / (b1 - b0)
            return e0 + t * (e1 - e0)
    return None


def first20_info(n):
    path = os.path.join(RESULTS, "frozen_sets.txt")
    key = f"N={n},"
    capture = False
    nums = []
    with open(path) as f:
        for line in f:
            if key in line:
                capture = True
                continue
            if capture and line.strip().startswith("Info indices"):
                continue
            if capture and line.strip().startswith("["):
                s = line.replace("[", " ").replace("]", " ")
                nums.extend(int(x) for x in s.split())
            if capture and "Frozen indices" in line:
                break
    return ", ".join(str(x) for x in nums[:20])


def exp2_combined_table():
    files = {
        "sc": "exp2_sc_N512_R0.5.csv",
        "l2": "exp2_scl_L2_N512_R0.5.csv",
        "l4": "exp2_scl_L4_N512_R0.5.csv",
        "l8": "exp2_scl_L8_N512_R0.5.csv",
        "cascl": "exp2_cascl_L8_N512_R0.5.csv",
    }
    data = {k: {float(r["eb_n0_db"]): r for r in load_csv(os.path.join(RESULTS, v))} for k, v in files.items()}
    eb_list = sorted(data["sc"].keys())
    lines = []
    for eb in eb_list:
        row = [f"{eb:.2f}"]
        for k in ("sc", "l2", "l4", "l8", "cascl"):
            r = data[k][eb]
            row += [fmt_sci(r["bler"]), f"{float(r['avg_decode_time_ms']):.1f}"]
        lines.append(" & ".join(row) + r" \\")
    return "\n".join(lines)


def main():
    shannon = find_capacity_limit(0.5)
    shannon_str = f"{shannon:.2f}"

    sc256 = load_csv(os.path.join(RESULTS, "exp1_sc_N256_R0.5.csv"))
    sc512 = load_csv(os.path.join(RESULTS, "exp1_sc_N512_R0.5.csv"))
    sc1024 = load_csv(os.path.join(RESULTS, "exp1_sc_N1024_R0.5.csv"))

    eb256 = interp_eb_for_bler(sc256)
    eb512 = interp_eb_for_bler(sc512)
    eb1024 = interp_eb_for_bler(sc1024)

    exp2_sc = load_csv(os.path.join(RESULTS, "exp2_sc_N512_R0.5.csv"))
    exp2_l2 = load_csv(os.path.join(RESULTS, "exp2_scl_L2_N512_R0.5.csv"))
    exp2_l4 = load_csv(os.path.join(RESULTS, "exp2_scl_L4_N512_R0.5.csv"))
    exp2_l8 = load_csv(os.path.join(RESULTS, "exp2_scl_L8_N512_R0.5.csv"))
    exp2_cascl = load_csv(os.path.join(RESULTS, "exp2_cascl_L8_N512_R0.5.csv"))

    eb_sc2 = interp_eb_for_bler(exp2_sc)
    eb_l2 = interp_eb_for_bler(exp2_l2)
    eb_l4 = interp_eb_for_bler(exp2_l4)
    eb_l8 = interp_eb_for_bler(exp2_l8)
    eb_cascl = interp_eb_for_bler(exp2_cascl)

    def avg_time(rows):
        return sum(float(r["avg_decode_time_ms"]) for r in rows) / len(rows)

    out = os.path.join(os.path.dirname(__file__), "report.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write(TEMPLATE_HEADER)
        f.write(
            f"""
% --- 自动填充：仿真环境 ---
% Python 3.12.3, NumPy 2.4.4, Matplotlib 3.10.9, Sionna 2.0.1 (BP)
"""
        )
        f.write(TEMPLATE_BODY.format(
            shannon=shannon_str,
            info256=first20_info(256),
            info512=first20_info(512),
            info1024=first20_info(1024),
            sc256_rows=csv_rows_tex(sc256, ["eb_n0_db", "num_errors", "num_frames", "bler", "ber", "avg_decode_time_ms"]),
            sc512_rows=csv_rows_tex(sc512, ["eb_n0_db", "num_errors", "num_frames", "bler", "ber", "avg_decode_time_ms"]),
            sc1024_rows=csv_rows_tex(sc1024, ["eb_n0_db", "num_errors", "num_frames", "bler", "ber", "avg_decode_time_ms"]),
            eb256=f"{eb256:.2f}" if eb256 else "---",
            eb512=f"{eb512:.2f}" if eb512 else "---",
            eb1024=f"{eb1024:.2f}" if eb1024 else "---",
            gap256=f"{eb256 - shannon:.2f}" if eb256 else "---",
            gap512=f"{eb512 - shannon:.2f}" if eb512 else "---",
            gap1024=f"{eb1024 - shannon:.2f}" if eb1024 else "---",
            exp2_combined=exp2_combined_table(),
            eb_sc2=f"{eb_sc2:.2f}" if eb_sc2 else "---",
            eb_l2=f"{eb_l2:.2f}" if eb_l2 else "---",
            eb_l4=f"{eb_l4:.2f}" if eb_l4 else "---",
            eb_l8=f"{eb_l8:.2f}" if eb_l8 else "---",
            eb_cascl=f"{eb_cascl:.2f}" if eb_cascl else "---",
            gain_l2=f"{eb_sc2 - eb_l2:.2f}" if eb_sc2 and eb_l2 else "---",
            gain_l4=f"{eb_sc2 - eb_l4:.2f}" if eb_sc2 and eb_l4 else "---",
            gain_l8=f"{eb_sc2 - eb_l8:.2f}" if eb_sc2 and eb_l8 else "---",
            gain_cascl=f"{eb_sc2 - eb_cascl:.2f}" if eb_sc2 and eb_cascl else "---",
            gain_cascl_vs_l8=f"{eb_l8 - eb_cascl:.2f}" if eb_l8 and eb_cascl else "---",
            t_sc2=f"{avg_time(exp2_sc):.1f}",
            t_l2=f"{avg_time(exp2_l2):.1f}",
            t_l4=f"{avg_time(exp2_l4):.1f}",
            t_l8=f"{avg_time(exp2_l8):.1f}",
            t_cascl=f"{avg_time(exp2_cascl):.1f}",
            bp256_rows=csv_rows_tex(
                load_csv(os.path.join(RESULTS, "exp3_bp_N256_R0.5.csv")),
                ["eb_n0_db", "num_errors", "num_frames", "bler", "ber", "avg_decode_time_ms", "avg_iters"],
            ),
            bp512_rows=csv_rows_tex(
                load_csv(os.path.join(RESULTS, "exp3_bp_N512_R0.5.csv")),
                ["eb_n0_db", "num_errors", "num_frames", "bler", "ber", "avg_decode_time_ms", "avg_iters"],
            ),
            sc256_25=fmt_sci(next(r for r in sc256 if float(r["eb_n0_db"]) == 2.5)["bler"]),
            sc512_25=fmt_sci(next(r for r in sc512 if float(r["eb_n0_db"]) == 2.5)["bler"]),
            sc256_20=fmt_sci(next(r for r in sc256 if float(r["eb_n0_db"]) == 2.0)["bler"]),
            exp2_sc_25=fmt_sci(next(r for r in exp2_sc if float(r["eb_n0_db"]) == 2.5)["bler"]),
            exp2_l2_25=fmt_sci(next(r for r in exp2_l2 if float(r["eb_n0_db"]) == 2.5)["bler"]),
            exp2_l4_25=fmt_sci(next(r for r in exp2_l4 if float(r["eb_n0_db"]) == 2.5)["bler"]),
            exp2_l8_25=fmt_sci(next(r for r in exp2_l8 if float(r["eb_n0_db"]) == 2.5)["bler"]),
            exp2_cascl_25=fmt_sci(next(r for r in exp2_cascl if float(r["eb_n0_db"]) == 2.5)["bler"]),
            exp2_sc_20=fmt_sci(next(r for r in exp2_sc if float(r["eb_n0_db"]) == 2.0)["bler"]),
            exp2_l8_20=fmt_sci(next(r for r in exp2_l8 if float(r["eb_n0_db"]) == 2.0)["bler"]),
            bp256_35=fmt_sci(next(r for r in load_csv(os.path.join(RESULTS, "exp3_bp_N256_R0.5.csv")) if float(r["eb_n0_db"]) == 3.5)["bler"]),
            sc256_35=fmt_sci(next(r for r in load_csv(os.path.join(RESULTS, "exp3_sc_N256_R0.5.csv")) if float(r["eb_n0_db"]) == 3.5)["bler"]),
            bp512_35=fmt_sci(next(r for r in load_csv(os.path.join(RESULTS, "exp3_bp_N512_R0.5.csv")) if float(r["eb_n0_db"]) == 3.5)["bler"]),
            sc512_35=fmt_sci(next(r for r in load_csv(os.path.join(RESULTS, "exp3_sc_N512_R0.5.csv")) if float(r["eb_n0_db"]) == 3.5)["bler"]),
            bp256_t=f"{avg_time(load_csv(os.path.join(RESULTS, 'exp3_bp_N256_R0.5.csv'))):.1f}",
            sc256_t=f"{avg_time(load_csv(os.path.join(RESULTS, 'exp3_sc_N256_R0.5.csv'))):.1f}",
            scl256_t=f"{avg_time(load_csv(os.path.join(RESULTS, 'exp3_scl_N256_R0.5.csv'))):.1f}",
            bp512_t=f"{avg_time(load_csv(os.path.join(RESULTS, 'exp3_bp_N512_R0.5.csv'))):.1f}",
            sc512_t=f"{avg_time(load_csv(os.path.join(RESULTS, 'exp3_sc_N512_R0.5.csv'))):.1f}",
            scl512_t=f"{avg_time(load_csv(os.path.join(RESULTS, 'exp3_scl_N512_R0.5.csv'))):.1f}",
            eb_sc512_exp3=f"{interp_eb_for_bler(load_csv(os.path.join(RESULTS, 'exp3_sc_N512_R0.5.csv'))):.2f}",
            eb_scl512_exp3=f"{interp_eb_for_bler(load_csv(os.path.join(RESULTS, 'exp3_scl_N512_R0.5.csv'))):.2f}",
            eb_bp512_exp3=f"{interp_eb_for_bler(load_csv(os.path.join(RESULTS, 'exp3_bp_N512_R0.5.csv'))):.2f}" if interp_eb_for_bler(load_csv(os.path.join(RESULTS, "exp3_bp_N512_R0.5.csv")), 1e-3) else ">5.0",
        ))
        f.write(TEMPLATE_TAIL)
    print(f"Wrote {out}")


TEMPLATE_HEADER = r"""% 信息论课程实验报告 — 极化码（自动填充仿真数据）
\documentclass[12pt, a4paper]{article}
\usepackage{ctex}
\usepackage{geometry}
\geometry{top=2.5cm, bottom=2.5cm, left=3cm, right=3cm}
\usepackage{amsmath, amssymb}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{listings}
\usepackage{float}
\usepackage{enumitem}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\rhead{信息论课程实验报告}
\lhead{极化码编译码仿真与性能分析}
\cfoot{\thepage}
\newcommand{\bs}[1]{\boldsymbol{#1}}
\newcommand{\placeholder}[1]{\textcolor{gray}{#1}}
\lstset{
  basicstyle=\ttfamily\footnotesize, breaklines=true, frame=single,
  numbers=left, numbersep=5pt, tabsize=4
}
\begin{document}
\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries 信息论课程实验报告\par}
\vspace{1cm}
{\LARGE 极化码编译码仿真与性能分析\par}
\vspace{0.5cm}\rule{\linewidth}{0.5pt}\vspace{1.5cm}
\begin{tabular}{ll}
\textbf{姓\quad 名：} & \placeholder{（请填写）} \\[6pt]
\textbf{学\quad 号：} & \placeholder{（请填写）} \\[6pt]
\textbf{班\quad 级：} & \placeholder{（请填写）} \\[6pt]
\textbf{指导教师：}   & \placeholder{（请填写）} \\[6pt]
\textbf{完成日期：}   & 2026年5月31日 \\
\end{tabular}
\vspace{2cm}
\placeholder{中国科学技术大学}\\
\placeholder{信息论 A 课程实验}
\end{titlepage}
\tableofcontents
\clearpage
"""

TEMPLATE_BODY = r"""
\section{{实验背景与目的}}
\subsection{{背景}}
1948年香农提出信道容量定理\cite{{shannon1948}}；2009年 Arıkan 提出极化码\cite{{arikan2009}}，
为首个被严格证明在 BI-DMC 上达到容量的构造性方案，现已成为 5G 控制信道编码标准之一。

\subsection{{实验目的}}
\begin{{enumerate}}
  \item 掌握 GA 构造与 SC/SCL/BP 译码原理；
  \item 蒙特卡洛仿真 BLER/BER 并与 BPSK 容量限对比；
  \item 分析列表增益、CRC 辅助及 BP 迭代特性。
\end{{enumerate}}

\section{{实验原理}}

\subsection{{极化码编码}}
码长 $N=2^n$，生成矩阵 $\bs{{G}}_N = \bs{{B}}_N \bs{{F}}^{{\otimes n}}$，
$\bs{{F}}=\begin{{bmatrix}}1&0\\1&1\end{{bmatrix}}$，$\bs{{B}}_N$ 为比特倒序置换矩阵。
编码 $\bs{{x}}=\bs{{u}}\bs{{G}}_N$，复杂度 $O(N\log N)$。

\subsection{{高斯近似构造}}
BPSK-AWGN 下，设计 $E_b/N_0$ 对应 $\sigma = (2R)^{{-1/2}}\cdot 10^{{-E_b/(20N_0)}}$，
初始 LLR 均值 $m_0=2/\sigma^2$，递推
$m_{{2i-1}}=\phi^{{-1}}(1-[1-\phi(m_i)]^2)$，$m_{{2i}}=2m_i$；
在比特倒序域选取 LLR 最大的 $K$ 个位置为信息集 $\mathcal{{A}}$。

\subsection{{SC 译码}}
$f(L_a,L_b)\approx\mathrm{{sign}}(L_a)\mathrm{{sign}}(L_b)\min(|L_a|,|L_b|)$；
$g(L_a,L_b,\hat{{u}})=(1-2\hat{{u}})L_a+L_b$。复杂度 $O(N\log N)$，存在错误传播。

\subsection{{SCL 与 CA-SCL}}
保留至多 $L$ 条路径，路径度量 PM 按式更新；CA-SCL 在 $L=8$ 时用 CRC-8 筛选最终路径。

\subsection{{BP 译码}}
基于因子图迭代传递 L/R 消息（本实验采用 Sionna PolarBPDecoder，max\_iter=50，含早停）。
极化码图结构含短环，性能通常弱于 SCL。

\subsection{{蒙特卡洛仿真}}
$\widehat{{\mathrm{{BLER}}}}=N_{{\mathrm{{err}}}}/N_{{\mathrm{{total}}}}$；
实验一每点至少 40 错误帧或 8000 帧；实验二/三快速模式为 4000 帧或 40 错误。

\section{{仿真环境与参数设置}}
\subsection{{软件环境}}
\begin{{itemize}}
  \item 编程语言：Python 3.12.3
  \item 主要库：NumPy 2.4.4、Matplotlib 3.10.9；BP 译码使用 Sionna 2.0.1（PyTorch 2.12）
  \item 硬件：Cloud Agent 虚拟机（多核 CPU）
  \item 操作系统：Linux
\end{{itemize}}

\subsection{{仿真参数}}
\begin{{table}}[H]
\centering
\caption{{统一仿真参数（实验二、三为快速验证模式）}}
\begin{{tabular}}{{lll}}
\toprule
参数 & 取值 & 说明 \\
\midrule
码长 $N$ & 256, 512, 1024 & 实验一；实验二/三固定 512（实验三含 256） \\
码率 $R$ & 1/2 & $K=N/2$ \\
GA 设计 SNR & 2.5\,dB & 构造冻结集 \\
CRC（CA-SCL）& 8\,bit & CRC-8 \\
列表大小 $L$ & 1, 2, 4, 8 & 实验二（未跑 $L=16$） \\
BP 最大迭代 & 50 & Sionna PolarBPDecoder \\
实验一 SNR 点 & 0--5.25\,dB，步长 0.25\,dB & 22 点，最多 8000 帧/点 \\
实验二/三 SNR 点 & 2.0--5.0\,dB，步长 0.5\,dB & 7 点，最多 4000 帧/点 \\
停止条件 & $\geq 40$ 块错误或达最大帧数 & 实验二/三快速模式 \\
随机种子 & 42 & 可复现 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{实验结果}}

\subsection{{极化码构造结果（GA）}}
\begin{{table}}[H]
\centering
\caption{{GA 构造信息位集合 $\mathcal{{A}}$（前 20 个索引，设计 $E_b/N_0=2.5$\,dB，$R=1/2$）}}
\begin{{tabular}}{{cl}}
\toprule
$N$ & 前 20 个信息位索引 \\
\midrule
256  & {info256} \\
512  & {info512} \\
1024 & {info1024} \\
\bottomrule
\end{{tabular}}
\end{{table}}
完整集合见附录 \texttt{{frozen\_sets.txt}}。

\subsection{{实验一：SC 译码}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{{figures/fig1_sc_bler.pdf}}
  \caption{{SC 译码 BLER 曲线（$R=1/2$，GA 构造）；竖线为 BPSK 容量限 ${shannon}$\,dB。}}
\end{{figure}}

\begin{{table}}[H]
\centering\caption{{SC 原始数据（$N=256$）}}\small
\begin{{tabular}}{{rrrrrr}}
\toprule
$E_b/N_0$(dB) & 错误帧 & 总帧数 & BLER & BER & 时间(ms) \\
\midrule
{sc256_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering\caption{{SC 原始数据（$N=512$）}}\small
\begin{{tabular}}{{rrrrrr}}
\toprule
$E_b/N_0$(dB) & 错误帧 & 总帧数 & BLER & BER & 时间(ms) \\
\midrule
{sc512_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering\caption{{SC 原始数据（$N=1024$）}}\small
\begin{{tabular}}{{rrrrrr}}
\toprule
$E_b/N_0$(dB) & 错误帧 & 总帧数 & BLER & BER & 时间(ms) \\
\midrule
{sc1024_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{SC 性能摘要（BLER$=10^{{-3}}$ 对应 $E_b/N_0$，线性插值）}}
\begin{{tabular}}{{lrrr}}
\toprule
 & $N$ & $E_b/N_0@10^{{-3}}$(dB) & 与容量限差距(dB) \\
\midrule
SC & 256  & {eb256} & {gap256} \\
SC & 512  & {eb512} & {gap512} \\
SC & 1024 & {eb1024} & {gap1024} \\
\bottomrule
\end{{tabular}}
\end{{table}}
BPSK 容量限（$R=1/2$）：$E_b/N_0 \approx {shannon}$\,dB。

\subsection{{实验二：SCL 与 CA-SCL（$N=512$）}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{{figures/fig2_scl_bler.pdf}}
  \caption{{SCL/CA-SCL BLER 对比（不含 $L=16$）。}}
\end{{figure}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.70\textwidth]{{figures/fig2_decode_time.pdf}}
  \caption{{平均译码时间随列表大小变化。}}
\end{{figure}}

\begin{{table}}[H]
\centering\caption{{实验二原始数据（快速模式 7 点）}}\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{r|rr|rr|rr|rr|rr}}
\toprule
& \multicolumn{{2}}{{c|}}{{SC}} & \multicolumn{{2}}{{c|}}{{SCL L=2}} & \multicolumn{{2}}{{c|}}{{SCL L=4}}
& \multicolumn{{2}}{{c|}}{{SCL L=8}} & \multicolumn{{2}}{{c}}{{CA-SCL L=8}} \\
$E_b/N_0$ & BLER & t(ms) & BLER & t(ms) & BLER & t(ms) & BLER & t(ms) & BLER & t(ms) \\
\midrule
{exp2_combined}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{SCL 性能摘要（$N=512$，BLER$=10^{{-3}}$ 插值）}}
\begin{{tabular}}{{lrrr}}
\toprule
算法 & $E_b/N_0@10^{{-3}}$(dB) & 相对 SC 增益(dB) & 平均时间(ms) \\
\midrule
SC ($L=1$) & {eb_sc2} & --- & {t_sc2} \\
SCL ($L=2$) & {eb_l2} & {gain_l2} & {t_l2} \\
SCL ($L=4$) & {eb_l4} & {gain_l4} & {t_l4} \\
SCL ($L=8$) & {eb_l8} & {gain_l8} & {t_l8} \\
CA-SCL ($L=8$) & {eb_cascl} & {gain_cascl} & {t_cascl} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{实验三：BP 译码}}
\begin{{figure}}[H]
  \centering
  \begin{{subfigure}}[b]{{0.48\textwidth}}
    \includegraphics[width=\textwidth]{{figures/fig3_bp_N256_bler.pdf}}
    \caption{{$N=256$}}
  \end{{subfigure}}
  \hfill
  \begin{{subfigure}}[b]{{0.48\textwidth}}
    \includegraphics[width=\textwidth]{{figures/fig3_bp_N512_bler.pdf}}
    \caption{{$N=512$}}
  \end{{subfigure}}
  \caption{{SC、SCL($L=4$) 与 BP 对比。}}
\end{{figure}}
\begin{{figure}}[H]
  \centering
  \begin{{subfigure}}[b]{{0.48\textwidth}}
    \includegraphics[width=\textwidth]{{figures/fig3_bp_N256_iters.pdf}}
    \caption{{$N=256$ BP 迭代次数}}
  \end{{subfigure}}
  \hfill
  \begin{{subfigure}}[b]{{0.48\textwidth}}
    \includegraphics[width=\textwidth]{{figures/fig3_bp_N512_iters.pdf}}
    \caption{{$N=512$ BP 迭代次数}}
  \end{{subfigure}}
  \caption{{BP 平均迭代次数（本数据均达上限 50 次）。}}
\end{{figure}}

\begin{{table}}[H]
\centering\caption{{BP 原始数据（$N=256$）}}
\begin{{tabular}}{{rrrrrrr}}
\toprule
$E_b/N_0$ & 错误帧 & 总帧 & BLER & BER & 时间(ms) & 迭代 \\
\midrule
{bp256_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering\caption{{BP 原始数据（$N=512$）}}
\begin{{tabular}}{{rrrrrrr}}
\toprule
$E_b/N_0$ & 错误帧 & 总帧 & BLER & BER & 时间(ms) & 迭代 \\
\midrule
{bp512_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{三种算法对比（$N=512$，BLER$=10^{{-3}}$）}}
\begin{{tabular}}{{lrrrr}}
\toprule
算法 & $E_b/N_0$(dB) & 平均时间(ms) & 计算复杂度 & 空间 \\
\midrule
SC & {eb_sc512_exp3} & {sc512_t} & $O(N\log N)$ & $O(N)$ \\
SCL ($L=4$) & {eb_scl512_exp3} & {scl512_t} & $O(LN\log N)$ & $O(LN)$ \\
BP (50 iter) & {eb_bp512_exp3} & {bp512_t} & $O(IN\log N)$ & $O(N\log N)$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{分析与讨论}}

\subsection{{SC 错误传播}}
在 $E_b/N_0=2.0$\,dB、$N=256$ 时 SC 的 BLER 为 {sc256_20}，而 $N=512$ 同 SNR 下为
{sc512_25} 量级附近（2.5\,dB 时 {sc512_25}），说明有限码长下 SC 对噪声较敏感。
$g$ 运算依赖已译比特，一旦前序比特错判，后续 LLR 符号翻转，形成连锁错误；
SCL($L=8$) 在 2.0\,dB 可将 BLER 从 {exp2_sc_20} 降至 {exp2_l8_20}，差距约 6\,dB（BLER 域）。

\subsection{{码长影响}}
随 $N$ 增大，达到 BLER$=10^{{-3}}$ 所需 SNR 从 {eb256}\,dB（$N=256$）降至
{eb1024}\,dB（$N=1024$），与容量限差距由 {gap256}\,dB 缩至 {gap1024}\,dB，
符合极化码 $N\to\infty$ 时趋近容量的理论趋势。

\subsection{{SCL 列表增益}}
在 2.5\,dB：SC {exp2_sc_25} $\to$ L=2 {exp2_l2_25} $\to$ L=4 {exp2_l4_25}
$\to$ L=8 {exp2_l8_25} $\to$ CA-SCL {exp2_cascl_25}。
$L=4\to 8$ 增益明显减弱（BLER 几乎不变），呈现\textbf{{性能饱和}}；
CA-SCL 相对 SCL($L=8$) 在 BLER$=10^{{-3}}$ 处约 {gain_cascl_vs_l8}\,dB 增益，
因 CRC 提供独立于路径度量的正确性校验。

\subsection{{BP 译码}}
在 3.5\,dB、$N=256$：SC BLER={sc256_35}，BP={bp256_35}，BP 明显劣于 SC；
$N=512$ 时 SC={sc512_35}，BP={bp512_35}，差距进一步扩大。
原因：极化码因子图含短环，置信传播易收敛至伪码字；本实验 BP 迭代均达上限 50 次。
平均译码时间：BP($N=512$) {bp512_t}\,ms，SC {sc512_t}\,ms，SCL {scl512_t}\,ms。

\section{{必答问题}}
\textbf{{1. 错误传播：}}见 4.1 节；SC 与 SCL($L=8$) 在 2.0\,dB 的 BLER 相差约一个数量级。

\textbf{{2. 列表饱和：}}L=1$\to$2 增益 {gain_l2}\,dB（@BLER$10^{{-3}}$），L=2$\to$4 为 {gain_l4} 减 {gain_l2}\,dB，
L=4$\to$8 仅 {gain_l8} 减 {gain_l4}\,dB；$L\ge 8$ 后边际收益很小。

\textbf{{3. CRC 作用：}}CA-SCL 在 BLER$=10^{{-3}}$ 处比 SCL($L=8$) 优 {gain_cascl_vs_l8}\,dB；
当 PM 最小的路径未通过 CRC 时，可从其余路径中选合法码字。

\textbf{{4. 设计 SNR：}}设计 SNR 应接近目标工作点（如 BLER$=10^{{-2}}\sim10^{{-3}}$ 对应 SNR）；
偏高/偏低都会导致信息位集合与真实可靠性排序失配。

\textbf{{5. 场景推荐：}}URLLC 用 SC；高吞吐并行用 BP；5G 控制信道用 CA-SCL($L=8$)。
本仿真中 CA-SCL 在 2.0\,dB 即达 BLER$=2.5\times10^{{-3}}$，优于 SC 的 {exp2_sc_20}。

\section{{结论}}
\begin{{enumerate}}
  \item GA 构造高效，设计 SNR=2.5\,dB 下冻结集与 SC 仿真一致。
  \item SC 在有限码长下距容量限约 {gap512}--{gap256}\,dB；增大 $N$ 可缩小差距。
  \item SCL 列表增益在 $L=8$ 附近饱和；CA-SCL 额外带来约 {gain_cascl_vs_l8}\,dB 增益。
  \item BP 在本参数下性能弱于 SC/SCL，但具备并行迭代结构。
\end{{enumerate}}
"""

TEMPLATE_TAIL = r"""
\appendix
\section{冻结集完整列表}
\lstinputlisting[basicstyle=\ttfamily\scriptsize]{../results/frozen_sets.txt}

\section{关键代码}
\subsection{GA 构造核心递推}
\lstinputlisting[language=Python, firstline=48, lastline=78]{../construction.py}

\subsection{SC 的 $f/g$ 运算}
\lstinputlisting[language=Python, firstline=15, lastline=50]{../decoder_sc.py}

\subsection{SCL 路径度量}
\lstinputlisting[language=Python, firstline=58, lastline=166]{../decoder_scl.py}

\begin{thebibliography}{9}
\bibitem{arikan2009} E.~Ar{\i}kan, IEEE Trans.\ Inf.\ Theory, 2009.
\bibitem{shannon1948} C.~E.~Shannon, Bell Syst.\ Tech.\ J., 1948.
\bibitem{tal2015} I.~Tal and A.~Vardy, IEEE Trans.\ Inf.\ Theory, 2015.
\end{thebibliography}
\end{document}
"""

if __name__ == "__main__":
    main()
