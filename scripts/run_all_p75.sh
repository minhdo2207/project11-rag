#!/usr/bin/env bash
# Chạy lại toàn bộ sáu cấu hình với trần đầu ra 75 token — đúng bằng cửa sổ chấm của bài.
# Sinh tuần tự trên cổng 11435 (GPU sinh); chấm chạy song song trên 11436 (GPU giám khảo).
set -u
cd "$(dirname "$0")/.."
MODEL=llama3:8b-instruct-fp16
JUDGE=gemma4:31b
N=1335
P=75
mkdir -p logs

# cấu hình : nguồn tài liệu : hậu tố tên file
JOBS=(
  "B0:snippet:none"
  "B1.1:snippet:snippet"
  "B1.2:snippet:snippet"
  "B2:snippet:chunk"
  "B3:snippet:chunk"
  "B1:text:text"
)
gen_file() {  # $1 = cấu hình, $2 = hậu tố
    echo "results/crag/gen_${1}_${MODEL//:/-}_${2}_seed1_p${P}.jsonl"
}

scoring_chain() {
    for job in "${JOBS[@]}"; do
        IFS=: read -r cfg src tag <<< "$job"
        f=$(gen_file "$cfg" "$tag")
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
for job in "${JOBS[@]}"; do
    IFS=: read -r cfg src tag <<< "$job"
    echo "[sinh] $cfg bắt đầu $(date +%H:%M)"
    python3 -m src.crag.generate --config "$cfg" --model "$MODEL" --ref-source "$src" \
        --num-predict "$P" --keep-threshold 0.05 --host http://127.0.0.1:11435 \
        >> "logs/p75_gen_${cfg}.log" 2>&1
    echo "[sinh] $cfg xong $(date +%H:%M)"
done
wait $SCORER
echo "[XONG] toàn bộ sáu cấu hình $(date +%H:%M)"
