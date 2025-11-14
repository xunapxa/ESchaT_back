FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY app.py excel_parser.py ./
COPY *.xlsx ./

# 포트 노출
EXPOSE $PORT

# 앱 실행
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}

