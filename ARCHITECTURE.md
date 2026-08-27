# 모델 파이프라인 구조

## 1. 목적

이 프로젝트는 `Qwen/Qwen2.5-3B-Instruct`를 이용해 수학 문제의 정수 정답을 예측하는 Kaggle 대회용 파이프라인이다. 대회 규칙에 따라 Qwen2.5-3B-Instruct만 베이스 모델로 사용하며, 모든 학습과 추론은 로컬 모델로 수행한다.

전체 과정은 다음 네 단계로 구성된다.

```text
검증된 자기 풀이 생성 → LoRA 학습 → 고정 검증 세트 평가 → 제출 파일 생성
```

## 2. 디렉터리 구성

```text
PythonProject/
├─ main.py                       # CLI 진입점
├─ math_challenge/
│  ├─ cli.py                     # 명령어 정의와 전체 실행 흐름
│  ├─ data.py                    # CSV 로딩, 오류 ID 제외, 검증 분할
│  ├─ prompting.py               # 수학 풀이 프롬프트
│  ├─ modeling.py                # Qwen 및 LoRA 어댑터 로딩
│  ├─ generation.py              # 배치 생성과 self-consistency 다수결
│  ├─ parsing.py                 # 모델 출력에서 최종 정수 추출
│  ├─ bootstrap.py               # 정답으로 검증된 자기 풀이 생성
│  └─ training.py                # BF16 LoRA 및 4-bit QLoRA 학습
├─ tests/
│  └─ test_core.py               # 핵심 로직 단위 테스트
├─ requirements.txt              # 고정 라이브러리 버전
├─ README.md                     # 설치 및 실행 방법
├─ ARCHITECTURE.md               # 현재 문서
└─ guideline/                    # 대회 원본 규칙과 데이터 설명
```

실행 중 생성되는 데이터와 가중치는 다음 위치를 사용한다.

```text
data/                            # 대회 CSV 파일
artifacts/
├─ pseudo_solutions.jsonl        # 정답으로 검증된 자기 풀이
├─ pseudo_solutions.jsonl.progress
├─ pseudo_solutions.meta.json
├─ qwen-math-lora/               # 학습된 LoRA 어댑터
└─ validation_predictions.jsonl  # 검증 세트 예측 결과
submission.csv                   # Kaggle 제출 파일
```

`data/`, `artifacts/`, `submission.csv`는 Git 추적 대상에서 제외된다.

## 3. 전체 데이터 흐름

```text
deep_chal_math_train.csv
          │
          ├─ train_filtered_ids.csv에 포함된 오류 문항 제외
          ├─ ID 해시 기반 90% 학습 / 10% 검증 분할
          │
          ▼
Qwen 베이스 모델의 자기 풀이 생성
          │
          ├─ 문항마다 여러 풀이 샘플링
          └─ 최종 정수가 학습 정답과 같은 풀이만 채택
          │
          ▼
BF16 LoRA 또는 4-bit QLoRA 학습
          │
          ├─ 검증된 풀이가 있으면 풀이 과정과 정답을 학습
          └─ 검증된 풀이가 없으면 정답 태그를 학습
          │
          ▼
검증 및 테스트 추론
          │
          ├─ 문항마다 여러 풀이 생성
          ├─ 각 출력에서 최종 정수 추출
          └─ 정수 후보 다수결
          │
          ▼
submission.csv (`id`, `answer`)
```

## 4. CLI 구조

`main.py`는 `math_challenge.cli.main()`을 호출하는 최소 진입점이다. 실제 기능은 다음 하위 명령으로 제공된다.

| 명령 | 역할 |
|---|---|
| `inspect` | 데이터 개수, 오류 ID 제외 결과, 검증 세트 크기 확인 |
| `bootstrap` | 베이스 Qwen으로 검증된 자기 풀이 생성 |
| `train` | BF16 LoRA 또는 4-bit QLoRA 학습 |
| `evaluate` | 고정 검증 세트에서 Exact Match 평가 |
| `predict` | 테스트 CSV를 추론해 `submission.csv` 생성 |

모든 단계는 `--model-path`로 베이스 모델을 지정할 수 있다. 지정하지 않으면 `QWEN_MODEL_PATH` 환경 변수와 표준 모델 폴더를 탐색한다.

## 5. 데이터 처리

### 5.1 입력 형식

학습 데이터는 다음 컬럼을 사용한다.

| 컬럼 | 설명 |
|---|---|
| `id` | 문항 고유 식별자 |
| `question` | 자연어 및 LaTeX 수학 문제 |
| `answer` | 정수 정답 |

테스트 데이터에서는 `answer`가 비어 있거나 없어도 된다.

### 5.2 오류 문항 제외

`train_filtered_ids.csv`에 포함된 ID는 학습과 검증에서 모두 제외한다. 필터 파일은 `--filter-ids`로 지정한다.

### 5.3 고정 검증 분할

문항 ID의 SHA-256 해시를 이용해 기본 10%를 검증 세트로 선택한다. 따라서 다음 특성을 가진다.

- 실행할 때마다 같은 문항이 검증 세트로 선택된다.
- 입력 행의 순서를 바꿔도 분할이 변하지 않는다.
- bootstrap 및 학습 단계와 동일한 검증 세트를 공유한다.

기본 학습에서는 검증 세트를 제외한다. 최종 모델을 만들 때만 `train --train-all`로 전체 학습 데이터를 사용한다.

## 6. 프롬프트와 출력 형식

시스템 프롬프트는 모델에 다음 행동을 요구한다.

1. 문제를 단계적으로 해결한다.
2. 계산과 조건을 다시 확인한다.
3. 최종 정답이 정수임을 지킨다.
4. 마지막 줄을 `<answer>INTEGER</answer>` 형식으로 끝낸다.

예시는 다음과 같다.

```text
문제 풀이 과정...
<answer>-1234</answer>
```

이 형식은 학습과 추론에서 동일하게 사용해 출력 형식의 불일치를 줄인다.

## 7. 검증된 자기 풀이 생성

학습 데이터에는 최종 정답만 있고 풀이 과정이 없으므로, 베이스 Qwen이 자체적으로 풀이를 생성한다. 기본적으로 문항마다 네 개의 풀이를 샘플링한다.

다음 조건을 만족하는 풀이만 학습 데이터로 채택한다.

```text
생성된 풀이에서 추출한 최종 정수 == train.csv의 정답
```

채택된 레코드는 다음 JSONL 형식으로 저장된다.

```json
{
  "id": "train-000001",
  "question": "문제 텍스트",
  "answer": 42,
  "solution": "풀이 과정...\n<answer>42</answer>"
}
```

검증 세트 문제는 기본적으로 자기 풀이 생성에서 제외된다. 처리 완료 ID는 별도 progress 파일에 기록하므로 작업을 중단한 후 같은 명령으로 재개할 수 있다. 처음부터 다시 생성하려면 `--overwrite`를 사용한다.

## 8. 모델 로딩

베이스 모델 탐색 우선순위는 다음과 같다.

1. `--model-path` 인자
2. `QWEN_MODEL_PATH` 환경 변수
3. 프로젝트 주변의 `models/`, `shared/` 폴더
4. Hugging Face 로컬 캐시

모델은 `local_files_only=True`로 로딩하므로 추론 중 인터넷을 사용하지 않는다. 기본 attention 구현은 PyTorch SDPA를 사용한다.

학습 모드는 다음 두 가지다.

| 모드 | 사용 방법 | 용도 |
|---|---|---|
| BF16 LoRA | 기본값 | A5000 24GB 권장 설정 |
| 4-bit QLoRA | `--quantization 4bit` | 메모리가 부족할 때 사용 |

학습 결과는 전체 베이스 모델이 아니라 LoRA 어댑터로 저장된다. 추론 시 베이스 Qwen에 어댑터를 로드한다.

## 9. LoRA 학습

LoRA는 Qwen의 attention과 MLP 선형 계층에 적용된다.

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

A5000 기본 하이퍼파라미터는 다음과 같다.

| 항목 | 기본값 |
|---|---:|
| Precision | BF16 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Sequence length | 2048 |
| Device batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Epoch | 2 |
| Learning rate | 1e-4 |
| Scheduler | Cosine |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

Gradient checkpointing을 사용해 activation 메모리를 줄인다. BF16 학습에서는 fused AdamW를 사용하고, 4-bit QLoRA에서는 paged 8-bit AdamW를 사용한다.

### 9.1 학습 예제 구성

- 검증된 자기 풀이가 있는 문항: 풀이 과정과 최종 정답을 함께 학습한다.
- 검증된 풀이가 없는 문항: `<answer>정수</answer>`만 학습한다.
- 쉬운 문항이 과도하게 반복되지 않도록 문항당 자기 풀이를 기본 최대 2개만 사용한다.

### 9.2 Loss 마스킹

시스템 프롬프트와 사용자 문제 토큰은 label을 `-100`으로 설정한다. 따라서 loss는 assistant의 풀이와 정답 부분에만 적용된다.

긴 문장은 최대 sequence length에 맞춰 자르되 다음을 우선 보존한다.

- 채팅 템플릿 앞부분
- 문제 텍스트의 뒷부분
- assistant completion의 마지막 부분과 정답 태그

## 10. 추론과 Self-Consistency

추론에서는 하나의 답만 생성하지 않고 문항마다 여러 풀이를 생성한다. 기본값은 다음과 같다.

| 항목 | 기본값 |
|---|---:|
| 문항당 생성 수 | 5 |
| Temperature | 0.7 |
| Top-p | 0.9 |
| 최대 생성 토큰 | 512 |
| 문항 배치 크기 | 4 |

각 출력에서 정수 후보를 추출한 뒤 가장 많이 등장한 정수를 최종 예측으로 선택한다. 동률이면 먼저 생성된 후보를 선택한다.

정수 추출 우선순위는 다음과 같다.

1. `<answer>...</answer>`
2. LaTeX `\boxed{...}`
3. `Final answer`, `Answer`, `정답`, `답` 뒤의 정수
4. 출력에 등장한 마지막 정수

쉼표가 포함된 정수와 음수를 정규화한다. 모든 출력에서 정수를 찾지 못한 경우 현재 fallback 값은 `0`이다.

## 11. 검증 전략

권장 실험 순서는 다음과 같다.

```text
1. bootstrap으로 학습 세트의 검증된 자기 풀이 생성
2. 검증 세트를 제외하고 LoRA 학습
3. 고정 검증 세트에서 Exact Match 측정
4. 하이퍼파라미터와 추론 설정 확정
5. --train-all로 전체 학습 데이터를 사용해 최종 어댑터 학습
6. 테스트 데이터 추론 및 submission.csv 생성
```

평가 지표는 대회와 동일한 정수 Exact Match Accuracy다. 검증 결과는 `validation_predictions.jsonl`에 정답, 예측, 후보 정수와 정오 여부를 저장한다.

## 12. 학습 산출물

LoRA 학습 결과 폴더에는 다음 파일이 저장된다.

```text
artifacts/qwen-math-lora/
├─ adapter_config.json
├─ adapter_model.safetensors
├─ tokenizer 관련 파일
├─ checkpoint-*/
└─ challenge_metadata.json
```

`challenge_metadata.json`에는 다음 재현 정보가 포함된다.

- 베이스 모델 경로
- 학습 CSV와 오류 ID 파일
- 자기 풀이 데이터 경로
- 학습 예제 수
- 검증 비율과 전체 학습 여부
- LoRA, batch, sequence length, optimizer 관련 하이퍼파라미터

## 13. 테스트 범위

현재 단위 테스트는 다음 핵심 동작을 확인한다.

- CSV 로딩과 제출 CSV 생성
- 오류 ID 파일 처리
- ID 기반 검증 분할의 안정성
- `<answer>` 및 `\boxed{}` 정수 추출
- 소수 오인식 방지
- 배치 생성 결과의 문항별 그룹화와 다수결
- 긴 학습 예제의 길이 제한
- 프롬프트 loss 마스킹과 completion 보존

테스트 실행 명령은 다음과 같다.

```bash
python -m unittest discover -s tests -v
```

## 14. 주요 설계 의도

이 구조의 핵심 설계 의도는 다음과 같다.

- 최종 정답만 있는 학습 데이터를 검증된 풀이 데이터로 확장한다.
- 잘못된 자기 풀이를 정답 비교로 제거한다.
- 학습과 추론에서 동일한 정답 태그를 사용한다.
- 고정 검증 세트로 실험 간 성능을 비교한다.
- self-consistency로 단일 생성의 불안정성을 줄인다.
- 베이스 모델과 LoRA 어댑터를 분리해 대회 규칙 준수와 재현성을 확보한다.
