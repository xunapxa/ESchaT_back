# 빌더 스테이지
FROM python:3.11-slim as builder

WORKDIR /app

# 시스템 의존성 설치 (한 번에 정리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 timeout 설정
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# torch CPU 버전 먼저 설치 (명시적 버전으로 안정성 확보)
RUN pip install --no-cache-dir --timeout=1000 \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# 최종 런타임 이미지
FROM python:3.11-slim

WORKDIR /app

# Python 패키지 복사 (전체 디렉토리 복사로 경로 문제 방지)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 파일 복사
COPY app.py excel_parser.py ./
COPY *.xlsx ./

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 포트 노출
EXPOSE $PORT

# 앱 실행
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
