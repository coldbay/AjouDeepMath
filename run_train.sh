#!/usr/bin/env bash
set -euo pipefail

# 필요하면 실행 전에 같은 이름의 환경 변수로 경로를 덮어쓸 수 있습니다.
# 예: MODEL_PATH=/mnt/models/Qwen2.5-3B-Instruct bash run_train.sh
MODEL_PATH="${MODEL_PATH:-~/shared/nvme1/bonghyun/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-dataset/deep_chal_math_train_filtered.csv}"
PSEUDO_DATA="${PSEUDO_DATA:-artifacts/pseudo_solutions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/qwen-math-lora}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export QWEN_MODEL_PATH="$MODEL_PATH"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "오류: Qwen 모델을 찾을 수 없습니다: $MODEL_PATH" >&2
  echo "MODEL_PATH를 config.json이 있는 Qwen2.5-3B-Instruct 폴더로 지정하세요." >&2
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "오류: 학습 CSV를 찾을 수 없습니다: $TRAIN_FILE" >&2
  exit 1
fi

if [[ ! -f "$PSEUDO_DATA" ]]; then
  echo "오류: 검증된 자기 풀이 파일을 찾을 수 없습니다: $PSEUDO_DATA" >&2
  echo "먼저 다음 명령으로 자기 풀이를 생성하세요:" >&2
  echo "$PYTHON_BIN main.py bootstrap --model-path '$MODEL_PATH' --train '$TRAIN_FILE' --output '$PSEUDO_DATA'" >&2
  exit 1
fi

echo "베이스 모델: $MODEL_PATH"
echo "학습 데이터: $TRAIN_FILE"
echo "자기 풀이: $PSEUDO_DATA"
echo "출력 폴더: $OUTPUT_DIR"

"$PYTHON_BIN" main.py train \
  --model-path "$MODEL_PATH" \
  --train "$TRAIN_FILE" \
  --pseudo-data "$PSEUDO_DATA" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
