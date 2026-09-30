#!/usr/bin/env bash
# Chạy các bậc B2–B5: sinh trên một GPU, chấm trên GPU còn lại, gối đầu nhau.
#
#   scripts/run_ladder.sh <keep_threshold> <gate_threshold> ["B2 B3 ..."]
#
# Chuỗi sinh chạy tuần tự trên cổng 11435 (GPU 0). Chuỗi chấm chạy song song trên
# cổng 11436 (GPU 1): bậc nào sinh xong thì chấm ngay bậc đó, khỏi phải chờ hết.
set -u
cd "$(dirname "$0")/.."
KEEP=${1:-0.10}
GATE=${2:-0.10}
CONFIGS=${3:-"B2 B3 B4 B5"}
MODEL=llama3:8b-instruct-fp16
JUDGE=gemma4:31b
N=1335
mkdir -p logs

gen_file() {  # $1 = tên bậc
    echo "results/crag/gen_$1_${MODEL//:/-}_chunk_seed1.jsonl"
}

# ---- chuỗi chấm: chờ từng file sinh đủ dòng rồi chấm ----
scoring_chain() {
    for cfg in $CONFIGS; do
        f=$(gen_file "$cfg")
        while [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$N" ]; do
            pgrep -f "src.crag.generate" >/dev/null || { sleep 30; \
                [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$N" ] && \
                { echo "[chấm] $cfg: chuỗi sinh đã dừng mà file chưa đủ, bỏ qua"; break; }; }
            sleep 30
        done
        [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$N" ] && continue
        echo "[chấm] bắt đầu $cfg $(date +%H:%M)"
        python3 -m src.crag.score --gen "$f" --mode crag --judge "$JUDGE" \
            --host http://127.0.0.1:11436 >> "logs/score_$cfg.log" 2>&1
        echo "[chấm] xong $cfg $(date +%H:%M)"
    done
}

scoring_chain &
SCORER=$!

for cfg in $CONFIGS; do
    echo "[sinh] bắt đầu $cfg $(date +%H:%M)"
    python3 -m src.crag.generate --config "$cfg" --model "$MODEL" \
        --host http://127.0.0.1:11435 --keep-threshold "$KEEP" --gate-threshold "$GATE" \
        >> "logs/gen_$cfg.log" 2>&1
    echo "[sinh] xong $cfg $(date +%H:%M)"
done

wait $SCORER
echo "[xong] toàn bộ bậc thang $(date +%H:%M)"
