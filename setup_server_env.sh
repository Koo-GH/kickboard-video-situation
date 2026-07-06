#!/bin/bash
# ================================================
# kickboard_env 서버 환경 자동 재구성 스크립트
# 사용법: bash setup_server_env.sh
# ================================================

set -e  # 에러 발생 시 즉시 중단

echo "=========================================="
echo " kickboard_env 환경 재구성 시작"
echo "=========================================="

# Step 1: conda 환경 생성
echo ""
echo "[1/5] conda 환경 생성 (python 3.11)..."
conda create -n kickboard_env python=3.11 -y

# Step 2: 환경 활성화 및 프로젝트 이동
echo ""
echo "[2/5] 패키지 설치..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kickboard_env
cd ~/kickboard-video-situation

# Step 3: 기본 requirements 설치
pip install -r requirements.txt

# Step 4: CUDA 지원 PyTorch 설치
echo ""
echo "[3/5] CUDA 지원 PyTorch 설치 (cu121)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 5: CUDA 인식 확인
echo ""
echo "[4/5] CUDA 인식 확인..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Step 6: 모델 다운로드
echo ""
echo "[5/5] 모델 다운로드..."
echo " - Qwen2.5-VL-7B-Instruct 다운로드 중..."
python scripts/download_model.py --model Qwen/Qwen2.5-VL-7B-Instruct

echo " - VideoLLaMA3-7B 다운로드 중..."
python scripts/download_model.py --model DAMO-NLP-SG/VideoLLaMA3-7B

echo ""
echo "=========================================="
echo " 환경 재구성 완료!"
echo " 실행: conda activate kickboard_env"
echo " 추론: python scripts/run_inference.py --model qwen2_5_vl ..."
echo "=========================================="
