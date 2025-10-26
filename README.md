# Swift Identifier Extractor - RunPod GPU

RunPod GPU 환경에서 학습된 LoRA 모델로 Swift 식별자를 추출하는 프로젝트

## ✨ 주요 특징

- ✅ **학습 형식 일치**: Alpaca 형식 (### Instruction / ### Input / ### Response)
- ✅ **해시 기반 체크포인트**: 동일한 입력은 자동으로 스킵
- 🔄 **안전한 중단/재개**: Ctrl+C로 중단해도 진행상황 보존
- 🚀 **GPU 가속**: CUDA 지원으로 빠른 추론
- 📊 **토큰 수 필터링**: 학습 시 사용한 토큰 제한 준수

## 📁 프로젝트 구조

```
swift-identifier-extractor/
├── setup.sh                  # 환경 설정 스크립트
├── run_inference.py          # 메인 추론 스크립트
├── requirements.txt          # Python 패키지
├── README.md                 # 이 파일
├── models/                   # 모델 파일 (업로드 필요)
│   ├── base_model.gguf       # Phi-3-mini-128k-instruct (GGUF)
│   └── lora.gguf             # 학습된 LoRA 어댑터 (GGUF)
├── dataset.jsonl             # 입력 데이터셋 (업로드 필요)
├── checkpoint/               # 진행상황 저장
│   └── processed.jsonl       # 처리 완료 항목 (해시 + 결과)
└── output/                   # 결과 출력
    ├── identifiers.txt       # 추출된 식별자 목록
    └── identifiers_summary.json
```

## 🚀 빠른 시작

### 1단계: 환경 설정

```bash
# 실행 권한 부여 및 설정 스크립트 실행
chmod +x setup.sh
bash setup.sh
```

### 2단계: 파일 업로드

1. **모델 파일** → `models/` 디렉토리
   - `base_model.gguf` (Phi-3-mini-128k-instruct GGUF 버전)
   - `lora.gguf` (학습된 LoRA 어댑터 GGUF 버전)

2. **데이터셋** → 프로젝트 루트
   - `dataset.jsonl` (instruction, input 포함)

### 3단계: 추론 실행

```bash
python run_inference.py
```

---

## 📊 입력 데이터 형식 (중요!)

### JSONL 형식 (Alpaca)
```jsonl
{"instruction": "You are an expert Swift code auditor...", "input": "### Swift Source Code:\n```swift\n...\n```\n\n### AST Symbol Information:\n...", "output": "{\"identifiers\": [\"id1\", \"id2\"]}"}
```

**필수 필드:**
- `instruction`: 모델에게 주는 지시사항 (학습 시 사용한 instruction과 동일)
- `input`: Swift 코드 + AST + Rule 정보
- `output`: (선택) 정답 레이블 (추론 시에는 무시됨)

**프롬프트 형식 (자동 생성):**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

이 형식은 학습 시 사용한 `format_example` 함수와 동일합니다.

---

## 🎛️ 사용법

### 기본 실행
```bash
python run_inference.py
```

### 커스텀 옵션
```bash
python run_inference.py \
  --dataset my_dataset.jsonl \
  --base_model models/phi-3-mini-128k-instruct.gguf \
  --lora models/phi3_lora_adapter-Q4_K_M.gguf \
  --output output/my_identifiers.txt \
  --ctx 12288 \
  --max_input_tokens 10500 \
  --gpu_layers -1
```

### 옵션 설명

| 옵션 | 설명 | 기본값 | 비고 |
|------|------|--------|------|
| `--dataset` | 입력 JSONL 파일 | `dataset.jsonl` | Alpaca 형식 |
| `--base_model` | 베이스 모델 경로 | `models/base_model.gguf` | Phi-3 GGUF |
| `--lora` | LoRA 어댑터 경로 | `models/lora.gguf` | 학습된 어댑터 |
| `--output` | 출력 파일 경로 | `output/identifiers.txt` | |
| `--ctx` | 컨텍스트 크기 | `12288` | 학습 시 사용 |
| `--max_input_tokens` | 최대 입력 토큰 | `10500` | 필터링 기준 |
| `--gpu_layers` | GPU 레이어 수 | `-1` (전체) | |
| `--reset` | 체크포인트 초기화 | `False` | |

---

## 🔐 해시 기반 체크포인트

### 작동 원리

1. **해시 생성**: `instruction + input`을 SHA-256 해싱
2. **자동 스킵**: 동일한 해시는 재처리하지 않음
3. **결과 캐싱**: 체크포인트에 해시 + 결과 저장

### 체크포인트 파일 형식

```jsonl
{"hash": "a1b2c3...", "result": {"identifiers": ["id1", "id2"]}}
{"hash": "d4e5f6...", "result": {"identifiers": ["id3", "id4"]}}
```

---

## 💾 중단 및 재개

### 자동 체크포인트
- 처리 완료된 항목은 `checkpoint/processed.jsonl`에 자동 저장
- **Ctrl+C로 중단해도 안전**
- 다음 실행 시 자동으로 이어서 처리

### 체크포인트 초기화
```bash
python run_inference.py --reset
```

---

## 📈 토큰 수 관리

### 학습 시 설정
- **max_length**: 12288 tokens
- **실제 데이터**: ~10500 tokens 이하로 필터링됨

### 추론 시 설정
- **n_ctx**: 12288 (입력 컨텍스트)
- **max_tokens**: 8192 (출력 생성)
- **입력 필터링**: ~10500 tokens 이하만 처리

### 토큰 수 확인
```python
# 대략적인 토큰 수 = 문자 수 / 4
approx_tokens = len(instruction + input_text) / 4
```

---

## 📤 출력 형식

### identifiers.txt
```
APIClient
NetworkManager
UserProfile
calculateScore
validateInput
```

### identifiers_summary.json
```json
{
  "total_items": 1234,
  "processed_items": 1234,
  "total_identifiers": 5678,
  "unique_identifiers": 3456,
  "output_file": "output/identifiers.txt"
}
```

---

## 🔧 GPU 설정

### NVIDIA GPU 확인
```bash
nvidia-smi
```

### GPU 메모리 최적화
```bash
# 모든 레이어를 GPU에 로드 (권장)
--gpu_layers -1

# 일부 레이어만 GPU에 로드
--gpu_layers 32
```

---

## ⚠️ 문제 해결

### CUDA 에러
```bash
# CUDA 버전 확인
nvidia-smi

# llama-cpp-python 재설치 (CUDA 지원)
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 메모리 부족
```bash
# GPU 레이어 수 줄이기
python run_inference.py --gpu_layers 24

# 또는 컨텍스트 크기 줄이기 (비권장)
python run_inference.py --ctx 8192
```

### 토큰 수 초과
```bash
# 최대 입력 토큰 줄이기
python run_inference.py --max_input_tokens 8000
```

---

## 📝 예시 워크플로우

```bash
# 1. 환경 설정
bash setup.sh

# 2. 파일 업로드 확인
ls models/
ls dataset.jsonl

# 3. 데이터셋 확인 (첫 줄 출력)
head -n 1 dataset.jsonl | python -m json.tool

# 4. 추론 실행
python run_inference.py

# 5. 결과 확인
cat output/identifiers.txt | wc -l
cat output/identifiers_summary.json

# 6. 결과 다운로드
# output/identifiers.txt를 로컬로 다운로드
```

---

## 🎯 학습 형식과의 일치

### 학습 시 사용한 `format_example` 함수
```python
def format_example(ex):
    inst = ex.get("instruction")
    inp = ex.get("input")
    out = ex.get("output")
    
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}<|endoftext|>"
    else:
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}<|endoftext|>"
```

### 추론 시 사용하는 `_format_prompt` 메서드
```python
def _format_prompt(self, instruction: str, input_text: str) -> str:
    inst = instruction.strip()
    inp = input_text.strip()
    
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{inst}\n\n### Response:\n"
```

✅ 완전히 동일한 형식!

---

## 💡 개발자 노트

- **Alpaca 형식**: instruction + input → response 구조
- **Stop tokens**: `<|endoftext|>`, `###` 사용
- **해시 체크포인트**: instruction + input 기반
- **토큰 필터링**: 학습 시 제한 준수

---

## 🤝 기여

이 프로젝트는 Swingft 프로젝트의 일부입니다.

---

## 📄 라이센스

MIT