원본 Qwen 모델에 학습한 artifacts/qwen-math-lora 어뎁터를 연결한 형태입니다.

(test_submission input 파일의 구분을 위해 파일명을 test_submission_input.csv로 수정했습니다.)

```text
  python main.py predict \
    --model-path [Qwen2.5-3B-Instruct 원본 모델 경로] \
    --adapter-path artifacts/qwen-math-lora \
    --test dataset/test_submission_input.csv \
    --num-samples 7 \
    --output test_submission.csv
```

