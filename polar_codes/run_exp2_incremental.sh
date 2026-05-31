#!/usr/bin/env bash
# 实验二分阶段运行，每完成一项即 commit + push
set -euo pipefail
cd "$(dirname "$0")"

export POLAR_QUICK=1
export POLAR_MIN_ERRORS=40
export POLAR_MAX_FRAMES=4000

BRANCH="cursor/-bc-cf063858-69e7-4071-862f-81496e933335-e493"
LOG="results/exp2_run_full.log"

push_stage() {
    local msg="$1"
    shift
    git add "$@"
    if git diff --cached --quiet; then
        echo "[skip push] 无新变更: $msg"
        return 0
    fi
    git commit -m "$msg"
    git push -u origin "$BRANCH"
    echo "[pushed] $msg"
}

run_stage() {
    local stage="$1"
    local msg="$2"
    shift 2
    echo ""
    echo "========== $(date -u '+%Y-%m-%d %H:%M:%S UTC') 开始: $stage =========="
    EXP2_STAGE="$stage" python3 -u run_exp2.py 2>&1 | tee -a "$LOG"
    push_stage "$msg" "$@"
    echo "========== 完成: $stage =========="
}

mkdir -p results
echo "实验二增量运行开始 $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG"

run_stage sc \
    "实验二 SC 基线 (N=512, 快速模式)" \
    results/exp2_sc_N512_R0.5.csv results/exp2_run_full.log

run_stage scl_L2 \
    "实验二 SCL L=2 (N=512, 快速模式)" \
    results/exp2_scl_L2_N512_R0.5.csv results/exp2_run_full.log

run_stage scl_L4 \
    "实验二 SCL L=4 (N=512, 快速模式)" \
    results/exp2_scl_L4_N512_R0.5.csv results/exp2_run_full.log

run_stage scl_L8 \
    "实验二 SCL L=8 (N=512, 快速模式)" \
    results/exp2_scl_L8_N512_R0.5.csv results/exp2_scl_N512_R0.5.csv results/exp2_run_full.log

run_stage scl_L16 \
    "实验二 SCL L=16 (N=512, 快速模式)" \
    results/exp2_scl_L16_N512_R0.5.csv results/exp2_run_full.log

run_stage cascl \
    "实验二 CA-SCL L=8 (N=512, 快速模式)" \
    results/exp2_cascl_L8_N512_R0.5.csv results/exp2_run_full.log

run_stage plots \
    "实验二 fig2 图表（含 L=16）" \
    results/fig2_scl_bler.png results/fig2_scl_bler.pdf \
    results/fig2_decode_time.png results/fig2_decode_time.pdf \
    results/exp2_run_full.log

echo ""
echo "========== 实验二全部完成 $(date -u '+%Y-%m-%d %H:%M:%S UTC') =========="
