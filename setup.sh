#!/bin/bash
# setup.sh - RunPod GPU 환경 설정 스크립트

echo "======================================"
echo "RunPod GPU 환경 설정 시작"
echo "======================================"

# 시스템 업데이트
echo "[1/5] 시스템 패키지 업데이트..."
apt-get update -qq

# Python 필수 패키지 설치
echo "[2/5] Python 패키지 설치..."
pip install --upgrade pip -q
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 -q
pip install tqdm psutil -q

# 디렉토리 생성
echo "[3/5] 작업 디렉토리 생성..."
mkdir -p models
mkdir -p output
mkdir -p checkpoint

# 권한 설정
echo "[4/5] 권한 설정..."
chmod +x run_inference.py

# CUDA 확인
echo "[5/5] CUDA 환경 확인..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not installed (optional)"

echo ""
echo "======================================"
echo "✅ 환경 설정 완료!"
echo "======================================"
echo ""
echo "다음 단계:"
echo "1. models/ 디렉토리에 base_model.gguf와 lora.gguf 업로드"
echo "2. dataset.jsonl 파일 업로드"
echo "3. python run_inference.py 실행"
echo ""