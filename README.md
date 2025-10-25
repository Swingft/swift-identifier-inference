# Swift Identifier Extractor - RunPod GPU

RunPod GPU 환경에서 Swift 소스코드 데이터셋으로부터 식별자를 추출하는 프로젝트

## 📁 프로젝트 구조

```
swift-identifier-extractor/
├── setup.sh                  # 환경 설정 스크립트
├── run_inference.py          # 메인 추론 스크립트
├── requirements.txt          # Python 패키지
├── README.md                 # 이 파일
├── models/                   # 모델 파일 (업로드 필요)
│   ├── base_model.gguf       # 베이스 모델
│   └── lora.gguf             # LoRA 어댑터
├── dataset.jsonl             # 입력 데이터셋 (업로드 필요)
├── checkpoint/               # 진행상황 저장
│   └── processed.txt         # 처리 완료 파일 목록
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
   - `base_model.gguf` (베이스 모델)
   - `lora.gguf` (LoRA 어댑터)

2. **데이터셋** → 프로젝트 루트
   - `dataset.jsonl`

### 3단계: 추론 실행

```bash
python run_inference.py
```

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
  --base_model models/phi-3-128k.gguf \
  --lora models/my_lora.gguf \
  --output output/my_identifiers.txt \
  --ctx 8192 \
  --gpu_layers -1
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dataset` | 입력 JSONL 파일 | `dataset.jsonl` |
| `--base_model` | 베이스 모델 경로 | `models/base_model.gguf` |
| `--lora` | LoRA 어댑터 경로 | `models/lora.gguf` |
| `--output` | 출력 파일 경로 | `output/identifiers.txt` |
| `--ctx` | 컨텍스트 크기 | `8192` |
| `--gpu_layers` | GPU 레이어 수 (-1 = 전체) | `-1` |
| `--reset` | 체크포인트 초기화 | `False` |

---

## 💾 중단 및 재개

### 자동 체크포인트
- 처리 완료된 파일은 `checkpoint/processed.txt`에 자동 저장
- **Ctrl+C로 중단해도 안전**
- 다음 실행 시 자동으로 이어서 처리

### 체크포인트 초기화
```bash
python run_inference.py --reset
```

---

## 📊 입력 데이터 형식

### JSONL 형식
```jsonl
{"filename": "Alamofire_Session.swift", "repo": "Alamofire", "code": "import Foundation\n...", "size": 12345}
{"filename": "Kingfisher_ImageCache.swift", "repo": "Kingfisher", "code": "import UIKit\n...", "size": 8900}
```

**필수 필드:**
- `filename`: 파일명
- `repo`: 레포지토리 이름
- `code`: Swift 소스코드

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
```python
# 모든 레이어를 GPU에 로드 (권장)
--gpu_layers -1

# 일부 레이어만 GPU에 로드
--gpu_layers 32
```

---

## 📈 성능 팁

1. **컨텍스트 크기**: 큰 파일이 많으면 `--ctx 16384` 사용
2. **GPU 레이어**: 메모리가 충분하면 `-1` (전체 로드)
3. **배치 처리**: 현재는 순차 처리 (안정성 우선)

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

# 또는 컨텍스트 크기 줄이기
python run_inference.py --ctx 4096
```

### 모델 로딩 실패
- 모델 파일 경로 확인
- 파일 권한 확인 (`chmod 644 models/*.gguf`)
- 충분한 디스크 공간 확인

---

## 📝 예시 워크플로우

```bash
# 1. 환경 설정
bash setup.sh

# 2. 파일 업로드 확인
ls models/
ls dataset.jsonl

# 3. 추론 실행
python run_inference.py

# 4. 결과 확인
cat output/identifiers.txt | wc -l
cat output/identifiers_summary.json

# 5. 결과 다운로드
# output/identifiers.txt 파일을 로컬로 다운로드
```

---

## 🔄 재실행 시나리오

### 시나리오 1: 중단 후 재개
```bash
# 그냥 다시 실행하면 자동으로 이어서 처리됨
python run_inference.py
```

### 시나리오 2: 처음부터 다시 시작
```bash
# 체크포인트 초기화
python run_inference.py --reset
```

### 시나리오 3: 다른 데이터셋으로 실행
```bash
# 새 데이터셋 + 새 출력 파일
python run_inference.py \
  --dataset new_dataset.jsonl \
  --output output/new_identifiers.txt
```

---

## 💡 개발자 노트

- **순차 처리**: 모델 안정성을 위해 현재는 순차 처리 (배치 처리는 향후 추가 가능)
- **체크포인트**: 파일 단위로 저장되므로 안전하게 중단 가능
- **중복 제거**: 실시간으로 중복이 제거되어 저장됨

---

## 📞 지원

문제가 발생하면:
1. `output/identifiers_summary.json` 확인
2. `checkpoint/processed.txt` 확인
3. 로그 메시지 확인