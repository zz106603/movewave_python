# --- 빌드베이스 ---
FROM python:3.10-slim

# --- 타임존 설정 (선택) ---
ENV TZ=Asia/Seoul

# --- 작업 디렉토리 설정 ---
WORKDIR /app

# --- 필수 패키지만 복사 (캐시 최적화용) ---
COPY requirements.txt .

# --- 패키지 설치 (빌드 캐시 사용 유리) ---
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- 소스 코드 복사 ---
COPY . .

# --- 모델 캐시 경로 미포함 ---

# --- 실행 ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
