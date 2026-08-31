# Qwen2.5-3B LLM Math Challenge


구성 요소와 데이터 흐름에 대한 자세한 설명은 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

- 베이스 모델은 **Qwen/Qwen2.5-3B-Instruct만** 사용합니다.
- 잘못된 train ID를 제외하고 고정 해시 기반 10% 검증 세트를 만듭니다.
- 베이스 모델이 생성한 풀이 중 학습 정답과 일치하는 풀이만 선별합니다.
- NVIDIA A5000 24GB에 맞춘 BF16 LoRA SFT를 수행합니다. 필요하면 4-bit QLoRA로 전환할 수 있습니다.
- 추론에서는 여러 풀이의 최종 정수를 다수결하고 `id,answer` 제출 파일을 만듭니다.

## 1. 준비

대회 파일을 다음처럼 두는 것을 권장합니다. 파일명은 달라도 CLI에 실제 경로를 주면 됩니다.

```text
data/
  deep_chal_math_train.csv
  train_filtered_ids.csv
  deep_chal_math_leaderboard_filtered.csv
```

CUDA가 구성된 A5000 환경의 가상환경에서 고정 버전 패키지를 설치합니다. `requirements.txt`의 PyTorch는 환경의 CUDA 드라이버와 맞는 공식 CUDA wheel로 설치해야 합니다.

```bash
python -m pip install -r requirements.txt
```

모델은 `--model-path`로 직접 지정하는 것이 가장 확실합니다. 매번 입력하지 않으려면 다음 환경 변수를 사용합니다.

```bash
export QWEN_MODEL_PATH=/shared/Qwen2.5-3B-Instruct
```

지정한 폴더에는 최소한 `config.json`, tokenizer 파일, `*.safetensors`가 있어야 합니다. 프로그램은 프로젝트 주변의 `models/`, `shared/`, Hugging Face 캐시도 탐색합니다.

## 2. 데이터 확인

```bash
python main.py inspect \
  --train data/deep_chal_math_train.csv \
  --filter-ids data/train_filtered_ids.csv
```

## 3. 검증된 자기 풀이 생성

학습 정답과 최종 답이 같은 풀이만 `JSONL`로 남깁니다. 별도 progress 파일도 기록하므로 중단 후 같은 명령을 실행하면 완료된 ID 다음부터 재개합니다. 처음부터 다시 생성하려면 `--overwrite`를 추가합니다.

```bash
python main.py bootstrap \
  --model-path /shared/Qwen2.5-3B-Instruct \
  --train data/deep_chal_math_train.csv \
  --filter-ids data/train_filtered_ids.csv \
  --samples-per-question 4 \
  --output artifacts/pseudo_solutions.jsonl
```

먼저 전체 동작만 확인하려면 `--max-rows 20`을 추가하세요. 기본값은 고정 검증 세트 10%를 제외합니다.

## 4. BF16 LoRA 학습

서버의 모델 경로가 `/shared/Qwen2.5-3B-Instruct`라면 제공된 실행 스크립트를 사용할 수 있습니다.

```bash
chmod +x run_train.sh
./run_train.sh
```

모델 경로가 다르면 파일을 수정하지 않고 환경 변수로 지정할 수 있습니다.

```bash
MODEL_PATH=/mnt/models/Qwen2.5-3B-Instruct ./run_train.sh
```

동일한 학습을 CLI에서 직접 실행하려면 다음 명령을 사용합니다.

```bash
python main.py train \
  --model-path /shared/Qwen2.5-3B-Instruct \
  --train data/deep_chal_math_train.csv \
  --filter-ids data/train_filtered_ids.csv \
  --pseudo-data artifacts/pseudo_solutions.jsonl \
  --output-dir artifacts/qwen-math-lora
```

A5000 24GB 기본값은 BF16, batch 2, gradient accumulation 8, sequence length 2048, LoRA rank 32입니다. 메모리가 부족하면 `--batch-size 1 --max-length 1536`을 먼저 적용하고, 그래도 부족할 때 `--quantization 4bit`를 사용하세요. 생성 단계에서 메모리가 부족하면 `--generation-batch-size`를 4에서 2 또는 1로 낮추면 됩니다.

## 5. 로컬 검증

```bash
python main.py evaluate \
  --model-path /shared/Qwen2.5-3B-Instruct \
  --adapter-path artifacts/qwen-math-lora \
  --train data/deep_chal_math_train.csv \
  --filter-ids data/train_filtered_ids.csv \
  --num-samples 5
```

빠른 비교에는 `--num-samples 1 --max-rows 100`을 사용합니다. 검증 성능을 확정한 뒤 최종 어댑터를 만들 때만 `train`에 `--train-all`을 추가해 전체 학습 데이터를 사용하세요.

## 6. 제출 파일 생성

```bash
python main.py predict \
  --model-path /shared/Qwen2.5-3B-Instruct \
  --adapter-path artifacts/qwen-math-lora \
  --test data/deep_chal_math_leaderboard_filtered.csv \
  --num-samples 5 \
  --output submission.csv
```

출력은 UTF-8 CSV이며 헤더는 `id,answer`, 모든 answer는 정수입니다. 최종 test 공개 시 `--test` 파일만 교체하면 됩니다. 추론은 `local_files_only=True`로 동작해 인터넷을 사용하지 않습니다.

## 재현성과 규칙 준수

어댑터 폴더에는 베이스 모델 경로, 데이터 목록, 주요 하이퍼파라미터를 담은 `challenge_metadata.json`이 함께 저장됩니다. 외부 공개 데이터를 추가로 사용할 경우 최종 방법론 문서에 반드시 출처와 접근 방법을 기록하세요. 테스트 문제를 외부 API나 검색 엔진에 보내면 대회 규칙 위반입니다.

테스트 실행:

```bash
python -m unittest discover -s tests -v
```
