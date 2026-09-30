#!/usr/bin/env bash
# Chạy hai bậc cuối: B4 (ràng buộc trích dẫn) và B5 (B4 + cổng từ chối).
# Cùng trần 75 token, cùng ngưỡng 0.05 như B3 — ngưỡng lấy từ src/crag/calibrate.py,
# đo trên sự hiện diện của đáp án trong đoạn, KHÔNG nhìn vào điểm của bậc nào.
# Sinh tuần tự trên 11435 (GPU sinh); chấm song song trên 11436 (GPU giám khảo).
set -u
cd "$(dirname "$0")/.."
MODEL=llama3:8b-instruct-fp16
JUDGE=gemma4:31b
N=1335
P=75
THR=0.05
mkdir -p logs

JOBS=(B4 B5)
gen_file() { echo "results/crag/gen_${1}_${MODEL//:/-}_chunk_seed1_p${P}.jsonl"; }

scoring_chain() {
    for cfg in "${JOBS[@]}"; do
        f=$(gen_file "$cfg")
        while [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$N" ]; do
            pgrep -f "src.crag.generate" >/dev/null || { sleep 45
                [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$N" ] && {
                    echo "[chấm] $cfg: chuỗi sinh đã dừng, bỏ qua"; break; }; }
            sleep 45
        done
        [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$N" ] && continue
        for mode in crag three; do
            echo "[chấm $mode] $cfg bắt đầu $(date +%H:%M)"
            python3 -m src.crag.score --gen "$f" --mode $mode --judge "$JUDGE" \
                --host http://127.0.0.1:11436 >> "logs/p75_score_${cfg}_${mode}.log" 2>&1
            echo "[chấm $mode] $cfg xong $(date +%H:%M)"
        done
    done
}

scoring_chain &
SCORER=$!
for cfg in "${JOBS[@]}"; do
    echo "[sinh] $cfg bắt đầu $(date +%H:%M)"
    python3 -m src.crag.generate --config "$cfg" --model "$MODEL" --ref-source snippet \
        --num-predict "$P" --keep-threshold "$THR" --gate-threshold "$THR" \
        --host http://127.0.0.1:11435 >> "logs/p75_gen_${cfg}.log" 2>&1
    echo "[sinh] $cfg xong $(date +%H:%M)"
done
wait $SCORER
echo "[XONG] B4 và B5 $(date +%H:%M)"
